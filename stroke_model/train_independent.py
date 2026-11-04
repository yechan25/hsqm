"""Train independent context masks on ACTUAL input-stroke annotations.

python -m stroke_model.train_independent --manifest data/manifest.json --init-checkpoint old.pt
No conversion from exclusive labels/template masks is performed: overlap labels
must be available in target_masks B,K,H,W. Manifest uses group_id/split/path.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm.auto import tqdm
from .reference_independent import IndependentStrokeModel,IndependentLossConfig,independent_stroke_loss
from .train_reference import read_manifest
from .reference_topology import TopologyStrokeModel, TopologyLossConfig, AlignmentConfig, topology_stroke_loss


class IndependentDataset(Dataset):
    def __init__(self,rows,size,shuffle_strokes=False,reference_jitter=0):
        self.rows,self.size=rows,size
        self.shuffle_strokes=shuffle_strokes
        self.reference_jitter=reference_jitter
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        with np.load(self.rows[index]['path'],allow_pickle=False) as arrays:
            if 'target_masks' not in arrays:
                raise ValueError('Need ACTUAL input target_masks; exclusive labels/template strokes cannot recover overlap annotations')
            image,reference,strokes,target=[torch.tensor(arrays[k].copy(),dtype=torch.float32)
                                           for k in ('image','reference','strokes','target_masks')]
            valid=torch.tensor(arrays['valid_pixels'].copy(),dtype=torch.float32) if 'valid_pixels' in arrays else torch.ones_like(image)
        if image.shape!=(1,self.size,self.size) or reference.shape!=image.shape or valid.shape!=image.shape:
            raise ValueError('image/reference/valid_pixels must be 1,S,S')
        if strokes.ndim!=3 or strokes.shape[-2:]!=image.shape[-2:] or not len(strokes) or target.shape!=strokes.shape:
            raise ValueError('strokes and actual target_masks must be matching K,S,S')
        for value in (image,reference,strokes,target,valid):
            if not torch.isfinite(value).all() or (value<0).any() or (value>1).any():raise ValueError('Arrays must be ink [0,1]')
        if not (strokes.flatten(1).amax(1)>.5).all():raise ValueError('Unpadded expected reference slots must be nonempty')
        if valid.sum()==0:raise ValueError('Need supervised pixels')
        if self.reference_jitter:
            dy,dx=torch.randint(-self.reference_jitter,self.reference_jitter+1,(2,)).tolist()
            def shift(value):
                result=torch.zeros_like(value)
                y0,y1=max(0,dy),self.size+min(0,dy)
                x0,x1=max(0,dx),self.size+min(0,dx)
                result[...,y0:y1,x0:x1]=value[...,y0-dy:y1-dy,x0-dx:x1-dx]
                return result
            moved=shift(strokes)
            if (moved.flatten(1).amax(1)>.5).all():
                reference,strokes=shift(reference),moved
        if self.shuffle_strokes:
            order=torch.randperm(len(strokes));strokes,target=strokes[order],target[order]
        return image,reference,strokes,target,valid


def collate_independent(samples):
    count=max(len(s[2]) for s in samples)
    pad=lambda t:torch.nn.functional.pad(t,(0,0,0,0,0,count-len(t)))
    return (torch.stack([s[0] for s in samples]),torch.stack([s[1] for s in samples]),
            torch.stack([pad(s[2]) for s in samples]),torch.stack([pad(s[3]) for s in samples]),torch.stack([s[4] for s in samples]))


@torch.no_grad()
def evaluate_independent(model,loader,device):
    total=0.;count=0;absent_fp=0;absent_count=0
    model.eval()
    for batch in loader:
        image,reference,strokes,target,valid=[x.to(device) for x in batch]
        output=model(image,reference,strokes,allow_untrained=True)
        pred=(output['probabilities']>=.5)*valid
        target=target*valid
        mass=target.sum((-2,-1));active=output['active'];present=(mass>0)&active
        dice=(2*(pred*target).sum((-2,-1))+1e-6)/(pred.sum((-2,-1))+mass+1e-6)
        total+=float(dice[present].sum());count+=int(present.sum())
        absent=(~present)&active
        absent_fp+=int(((pred.sum((-2,-1))>0)&absent).sum());absent_count+=int(absent.sum())
    return {'present_macro_dice':total/max(count,1),'present_strokes':count,
            'absent_false_positive_rate':absent_fp/max(absent_count,1),'absent_strokes':absent_count}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--init-checkpoint',help='Optional V2 feature warm start, NOT independent inference weights')
    parser.add_argument('--image-size',type=int,default=128)
    parser.add_argument('--width',type=int,default=64)
    parser.add_argument('--output',default='outputs/reference_independent')
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--context-width',type=int,default=16)
    parser.add_argument('--model',choices=('independent','topology'),default='independent')
    parser.add_argument('--alignment-config',help='JSON overriding AlignmentConfig for topology model')
    parser.add_argument('--lr',type=float,default=2e-4)
    parser.add_argument('--loss-config',help='JSON overriding IndependentLossConfig')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--reference-jitter',type=int,default=2)
    parser.add_argument('--no-shuffle-strokes',action='store_true')
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.lr<=0 or args.reference_jitter<0:raise ValueError('Invalid training settings')
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    state=torch.load(args.init_checkpoint,map_location='cpu',weights_only=True) if args.init_checkpoint else None
    size,width=(state['image_size'],state['width']) if state else (args.image_size,args.width)
    if state and state.get('architecture', 'reference_v2') != 'reference_v2':
        raise ValueError('--init-checkpoint must be a compatible V2 checkpoint')
    topology=args.model=='topology'
    loss_type=TopologyLossConfig if topology else IndependentLossConfig
    loss_config=loss_type(**(json.loads(Path(args.loss_config).read_text()) if args.loss_config else {}))
    alignment_config=AlignmentConfig(**(json.loads(Path(args.alignment_config).read_text()) if args.alignment_config else {}))
    if args.alignment_config and not topology:raise ValueError('--alignment-config requires --model topology')
    rows=read_manifest(args.manifest);loaders={}
    for split in ('train','val'):
        dataset=IndependentDataset([row for row in rows if row['split']==split],size,
                                   shuffle_strokes=split=='train' and not args.no_shuffle_strokes,
                                   reference_jitter=args.reference_jitter if split=='train' else 0)
        if not len(dataset):raise ValueError('Missing split: '+split)
        loaders[split]=DataLoader(dataset,batch_size=args.batch_size,shuffle=split=='train',collate_fn=collate_independent)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_type=TopologyStrokeModel if topology else IndependentStrokeModel
    extra={'alignment_config':alignment_config} if topology else {}
    model=model_type(image_size=size,width=width,context_width=args.context_width,pretrained=state is None,**extra)
    if state:
        model.warm_start(state['model'])
    model=model.to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr)
    destination=Path(args.output);destination.mkdir(parents=True,exist_ok=True)
    (destination/'run_config.json').write_text(json.dumps({**vars(args),'loss':asdict(loss_config),
        'alignment':asdict(alignment_config) if topology else None},indent=2),encoding='utf-8')
    history=[];best=-1
    for epoch in range(args.epochs):
        model.train();running=0;seen=0;components={}
        for batch in tqdm(loaders['train'],desc=f'Epoch {epoch+1}/{args.epochs}',mininterval=5):
            image,reference,strokes,target,valid=[x.to(device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            loss_fn=topology_stroke_loss if topology else independent_stroke_loss
            losses=loss_fn(model(image,reference,strokes),target,image,valid,loss_config)
            if not torch.isfinite(losses['loss']):
                raise FloatingPointError('Nonfinite training loss; check annotations and settings')
            for key,value in losses.items():
                components[key]=components.get(key,0.)+float(value.detach())*len(image)
            losses['loss'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step();running+=float(losses['loss'].detach())*len(image);seen+=len(image)
        validation=evaluate_independent(model,loaders['val'],device)
        if validation['present_strokes']==0:raise ValueError('Validation needs actual present-stroke annotations')
        row={'epoch':epoch+1,'train_loss':running/max(seen,1),
             'train_components':{k:v/max(seen,1) for k,v in components.items()},
             **validation};history.append(row);print(json.dumps(row),flush=True)
        if validation['present_macro_dice']>best:
            best=validation['present_macro_dice']
            torch.save({'model':model.state_dict(),'architecture':model.architecture,'image_size':size,'width':width,
                        'context_width':args.context_width,'epoch':epoch+1,'supervision':'actual_input_multihot',
                        'loss_config':asdict(loss_config),'validation':validation,
                        'alignment_config':asdict(alignment_config) if topology else None},destination/'best.pt')
        (destination/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')


if __name__=='__main__':main()
