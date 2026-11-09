"""Gate A detector training only. No diffusion model or proposal training."""
from dataclasses import asdict
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .error_localizer import (StrokeErrorLocalizer, ErrorLocalizerConfig,
                             FrozenExtractorErrorLocalizer, error_localizer_loss, ARCHITECTURE)
from .error_synthesis import collate_errors


def _forward(model,batch,device):
    b={k:v.to(device) for k,v in batch.items()}
    output=model(b['rgb'],b['reference'],b['strokes'],b['codepoint'],b['active'])
    loss=error_localizer_loss(output,b['extra'],b['valid'],b['kind'])
    return b,output,loss


@torch.no_grad()
def evaluate_errors(model,dataset,device,batch_size=8,threshold=.5):
    model.eval();total=0.;count=0;tp=fp=fn=clean_fp=clean_pixels=0
    for batch in DataLoader(dataset,batch_size=batch_size,collate_fn=collate_errors):
        b,out,loss=_forward(model,batch,device)
        n=len(b['rgb']);total+=float(loss['loss'])*n;count+=n
        pred=out['extra_probability']>=threshold;true=b['extra']>.5;valid=b['valid']>.5
        tp+=int((pred&true&valid).sum());fp+=int((pred&~true&valid).sum());fn+=int((~pred&true&valid).sum())
        clean=(true&valid).flatten(1).sum(1)==0
        clean_fp+=int((pred[clean]&valid[clean]).sum());clean_pixels+=int(valid[clean].sum())
    if not count:raise ValueError('Empty evaluation dataset')
    return dict(loss=total/count,pixel_precision=tp/max(tp+fp,1),pixel_recall=tp/max(tp+fn,1),
        pixel_iou=tp/max(tp+fp+fn,1),clean_false_positive_pixel_rate=clean_fp/max(clean_pixels,1),
        threshold=threshold,threshold_calibrated=False,samples=count)


def fit_error_localizer(extractor,train_set,val_set,output,*,device='cuda',epochs=10,
                        batch_size=8,learning_rate=2e-4,width=32,seed=42,metadata=None):
    if epochs<1 or batch_size<1 or learning_rate<=0:raise ValueError('Invalid training settings')
    destination=Path(output)
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    torch.manual_seed(seed);device=torch.device(device)
    head=StrokeErrorLocalizer(ErrorLocalizerConfig(width)).to(device)
    model=FrozenExtractorErrorLocalizer(extractor.to(device),head)
    optimizer=torch.optim.AdamW(head.parameters(),lr=learning_rate)
    loader=DataLoader(train_set,batch_size=batch_size,shuffle=True,collate_fn=collate_errors,
                      generator=torch.Generator().manual_seed(seed),num_workers=0)
    meta=dict(metadata or {},architecture=ARCHITECTURE,config=asdict(head.config),epochs=epochs,
              seed=seed,batch_size=batch_size,learning_rate=learning_rate,scope='synthetic detector Gate A only')
    (destination/'training_config.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    best=float('inf');history=[]
    for epoch in range(epochs):
        train_set.epoch=epoch;model.train();running=0.;count=0
        for batch in tqdm(loader,desc=f'epoch {epoch+1}/{epochs}'):
            optimizer.zero_grad(set_to_none=True)
            _,out,loss=_forward(model,batch,device)
            if not torch.isfinite(loss['loss']):raise FloatingPointError('Nonfinite training loss')
            loss['loss'].backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            optimizer.step();running+=float(loss['loss'].detach())*len(batch['rgb']);count+=len(batch['rgb'])
        val=evaluate_errors(model,val_set,device,batch_size)
        row=dict(epoch=epoch+1,train_loss=running/max(count,1),validation=val);history.append(row)
        print(json.dumps(row),flush=True)
        state=dict(architecture=ARCHITECTURE,config=asdict(head.config),model=head.state_dict(),
                   epoch=epoch+1,validation=val,threshold_calibrated=False,metadata=meta)
        torch.save(state,destination/'last.pt')
        if val['loss']<best:best=val['loss'];torch.save(state,destination/'best.pt')
        (destination/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    checkpoint=torch.load(destination/'best.pt',map_location=device,weights_only=True)
    head.load_state_dict(checkpoint['model'],strict=True);model.eval()
    return model,history
