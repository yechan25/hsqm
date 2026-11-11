"""Reference-path insertion, settling, then re-extracted local erasure.

Only missing structural ink is inserted in clean-image space. VAE round-trip
coverage is checked; unresolved restoration never enables the erasure phase.
"""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import affine_transform, distance_transform_edt, binary_dilation, convolve, label
from skimage.morphology import skeletonize

from diffusion_model.fitted_candidate_diffusion import InkConstraintObjective, CandidateObjective


def centroid(mask):
    yy, xx = np.indices(mask.shape)
    mass = max(float(mask.sum()), 1e-6)
    return np.array([(xx*mask).sum()/mass, (yy*mask).sum()/mass])


def align_prototypes(record):
    """Visible fitted anchors position missing prototypes; absent ink is not fit."""
    source, target = [], []
    masks = record.get('extracted_masks', record['fitted'])
    metadata = record.get('fit_metadata', [{} for _ in masks])
    for original, fitted, observed, info in zip(record['strokes'], record['fitted'], masks, metadata):
        if float(observed.max()) < .3 or info.get('warnings') or 'fallback' in info.get('method', ''):continue
        source.append(centroid(original));target.append(centroid(fitted))
    matrix = np.c_[np.eye(2), np.zeros(2)]
    kind = 'canonical_fallback_no_reliable_visible_anchors'
    if source:
        x, y = np.array(source), np.array(target)
        shift = np.median(y-x, axis=0).clip(-12, 12)
        matrix[:, 2] = shift;kind = 'visible_anchor_translation'
        if len(x) >= 3 and np.linalg.matrix_rank(np.c_[x, np.ones(len(x))]) == 3:
            fitted, *_ = np.linalg.lstsq(np.c_[x, np.ones(len(x))], y, rcond=None)
            candidate = fitted.T
            singular = np.linalg.svd(candidate[:, :2], compute_uv=False)
            residual = np.linalg.norm(np.c_[x, np.ones(len(x))]@fitted-y, axis=1)
            corners=np.array([[0,0],[0,record['strokes'].shape[-2]-1],
                              [record['strokes'].shape[-1]-1,0],
                              [record['strokes'].shape[-1]-1,record['strokes'].shape[-2]-1]])
            movement=np.linalg.norm(np.c_[corners,np.ones(4)]@fitted-corners,axis=1).max()
            if singular.min() >= .8 and singular.max() <= 1.25 and np.median(residual) < 3 and movement <= 16:
                matrix = candidate;kind = 'bounded_visible_anchor_affine'
    swap = np.array([[0, 1], [1, 0]])
    inverse = np.linalg.inv(matrix[:, :2])
    warped = np.stack([affine_transform(s, swap@inverse@swap, offset=-swap@inverse@matrix[:, 2],
                      output_shape=s.shape, order=1, mode='constant') for s in record['strokes']])
    return warped.clip(0, 1), dict(method=kind, matrix=matrix.tolist(), visible_anchors=len(source))


class RestoreSettleErase(InkConstraintObjective):
    def __init__(self, source, extractor, records, *, coverage_threshold=.85,
                 max_cycles=2, settle_steps=2):
        if len(records) != 4 or source.shape[:2] != (1, 3):raise ValueError('Four slots and one RGB sheet required')
        if not .5 < coverage_threshold <= 1 or max_cycles < 1 or settle_steps < 1:raise ValueError('Invalid cycle schedule')
        self.source = source.detach();self.extractor = extractor.eval().requires_grad_(False)
        self.threshold = coverage_threshold;self.max_cycles = max_cycles;self.settle_steps = settle_steps
        self.phase = 'restore';self.cycles = 1;self.settle_left = settle_steps
        self.background_ink = .015;self.required_ink = .7;self.preserve_weight = .5
        self.records = [];self.coverage_log = [];self.history = []
        self.erase = torch.zeros_like(self.ink(source));self.restore = torch.zeros_like(self.erase)
        self.required_paths = [];self.initial_missing = []
        h, w = source.shape[-2:]
        boxes = [(0,0,w//2,h//2),(w//2,0,w,h//2),(0,h//2,w//2,h),(w//2,h//2,w,h)]
        for slot, (raw, box) in enumerate(zip(records, boxes)):
            prototypes, alignment = align_prototypes(raw)
            size = raw['strokes'].shape[-2:]
            observed = self.crop(self.ink(source), box, size)[0, 0].cpu().numpy()
            proto_paths = [skeletonize(p > .2) for p in prototypes]
            visible_widths = []
            for mask, path in zip(raw['fitted'], proto_paths):
                values = distance_transform_edt(mask > .35)[path & (observed > .35)]
                if values.size >= 4:visible_widths.append(float(np.median(values)))
            borrowed_half = float(np.clip(np.median(visible_widths) if visible_widths else 2., 1.25, 5.))
            distances = [];radii = [];identities = []
            for k, (prototype_path, fitted) in enumerate(zip(proto_paths, raw['fitted'])):
                if not prototype_path.any():raise ValueError(f'Slot {slot+1} stroke {k+1}: empty prototype')
                prototype_coverage = float((observed[prototype_path] > .35).mean())
                info = raw.get('fit_metadata', [{} for _ in proto_paths])[k]
                unreliable = prototype_coverage < .35 or bool(info.get('warnings')) or 'fallback' in info.get('method', '')
                path = prototype_path if unreliable else skeletonize(fitted > .35)
                if not path.any():path = prototype_path
                distance = distance_transform_edt(~path)
                # Present strokes keep their original caps/serifs. Small endpoint
                # offsets are not missing structure. Whole absent strokes retain caps.
                if prototype_coverage >= .2:
                    degree=convolve(path.astype(np.int32),np.ones((3,3),np.int32),mode='constant')-path
                    ends=path&(degree==1)
                    if ends.any():
                        interior=path&(distance_transform_edt(~ends)>2.)
                        if interior.any():path=interior
                values = distance_transform_edt(fitted > .35)[path & (observed > .35)]
                half = float(np.clip(np.median(values) if values.size >= 4 else borrowed_half, 1.25, 5.))
                # A narrow, renderable band supplies minimum structure, not a full silhouette.
                missing = path & (distance_transform_edt(~(observed > .3)) > .75)
                if prototype_coverage >= .2:
                    components,count=label(missing,np.ones((3,3)))
                    minimum=max(2,int(np.ceil(path.sum()*.03)))
                    missing=np.isin(components,[j for j in range(1,count+1) if (components==j).sum()>=minimum])
                scaffold = binary_dilation(missing, iterations=max(1, int(round(half*.4))))
                canvas = torch.zeros_like(self.restore);self.place(canvas, self.tensor(scaffold), box)
                self.restore = torch.maximum(self.restore, canvas)
                core = torch.zeros_like(self.restore);self.place(core, self.tensor(path), box)
                core = (core > .2).float()
                self.required_paths.append(core)
                self.initial_missing.append(dict(slot=slot+1, stroke=k+1,
                    coverage=prototype_coverage, missing_path_pixels=int(missing.sum()),
                    entire_stroke_missing=prototype_coverage < .2, prototype_fallback=unreliable))
                distances.append(distance);radii.append(half);identities.append(k+1)
            self.records.append(dict(box=box,size=size,reference=self.tensor(raw['reference']).reshape(1,1,*size),
                strokes=torch.as_tensor(raw['strokes'],device=source.device,dtype=torch.float32)[None],
                distances=np.stack(distances),radii=np.asarray(radii)[:,None,None],
                alignment=alignment,stroke_ids=identities))
        self.restore = (self.restore > .2).float()
        self.path_union = torch.stack(self.required_paths).amax(0)
        self.initial_restore = self.restore.clone()
        self.update_support()
        self.fixed_preserve = self.preserve.clone()
        self.metadata = dict(kind='restore_settle_erase', max_cycles=max_cycles, settle_steps=settle_steps,
            coverage_threshold=coverage_threshold, missing_strokes=self.initial_missing,
            alignments=[r['alignment'] for r in self.records], candidate_rgb_matching=False,
            structural_ink_projection=True, restore_before_erase=True)

    def tensor(self, array):return torch.as_tensor(np.asarray(array).copy(),device=self.source.device,dtype=torch.float32)[None,None]
    @staticmethod
    def crop(image, box, size):
        x0,y0,x1,y1=box
        return F.interpolate(image[...,y0:y1,x0:x1],size,mode='bilinear',align_corners=False)
    @staticmethod
    def place(canvas, patch, box):
        x0,y0,x1,y1=box
        canvas[...,y0:y1,x0:x1]=F.interpolate(patch,(y1-y0,x1-x0),mode='bilinear',align_corners=False)

    def update_support(self):
        self.edit=torch.maximum(self.erase,self.restore)
        self.preserve=1-F.max_pool2d((self.edit>.01).float(),5,1,2)

    @torch.no_grad()
    def coverage(self, image):
        ink=self.ink(image)
        return [float(((ink>.5).float()*p).sum()/p.sum().clamp_min(1)) for p in self.required_paths]

    @torch.no_grad()
    def insertion_image(self, image):
        """Explicit minimum-ink projection inside the missing band only."""
        deficiency=F.relu(self.required_ink-self.ink(image))*(self.restore>.01).float()
        return (image-deficiency).clamp(0,1)

    @torch.no_grad()
    def project_clean(self, clean, decode, encode, rounds=3):
        current=clean.detach().float();before=self.coverage(decode(current))
        support=F.adaptive_max_pool2d((self.restore>.01).float(),clean.shape[-2:])
        support=F.max_pool2d(support,3,1,1)
        attempts=0
        for _ in range(rounds):
            image=decode(current)
            if min(self.coverage(image))>=self.threshold:break
            # A path newly lost after deletion is also eligible for restoration.
            target=self.insertion_image(image)
            projected=encode(target).detach().float()
            if not torch.isfinite(projected).all():raise FloatingPointError('Nonfinite structural projection')
            current=current+support*(projected-current)
            attempts+=1
        after=self.coverage(decode(current))
        return current,dict(projection_rounds=attempts,coverage_before=before,coverage_after=after,
            restoration_verified=min(after)>=self.threshold)

    @torch.no_grad()
    def refresh_erase(self, image):
        erase=torch.zeros_like(self.erase)
        for r in self.records:
            ink=self.crop(self.ink(image).clamp(0,1),r['box'],r['size'])
            p=self.extractor(ink,r['reference'],r['strokes'])['probabilities'][0].float().cpu().numpy()
            if not np.isfinite(p).all() or p.shape!=r['distances'].shape:raise ValueError('Invalid stroke extraction')
            # Protect complete expected stroke corridors, including newly inserted paths.
            outside=np.clip((r['distances']-r['radii']-1.5)/1.5,0,1)
            safe=(r['distances']<=r['radii']+1.).any(0)
            responsibility=(p*outside).sum(0)
            observed=ink[0,0].cpu().numpy()
            seeds=(observed>.15)&(responsibility>.5)&~safe
            weight=binary_dilation(seeds,iterations=1)*responsibility*(~safe)
            self.place(erase,self.tensor(weight),r['box'])
        self.erase=erase*(self.restore<=0).float()*(1-self.path_union)
        self.update_support()

    @torch.no_grad()
    def begin_step(self, image, index):
        cov=self.coverage(image)
        if min(cov)<self.threshold and self.phase=='restore':
            missing=self.path_union*(self.ink(image)<.5).float()
            self.restore=torch.maximum(self.restore,F.max_pool2d(missing,3,1,1))
        if min(cov)<self.threshold and self.phase!='restore':
            if self.cycles<self.max_cycles:
                self.cycles+=1;self.phase='restore'
                # Enlarge the restoration band only around paths that were lost.
                missing=self.path_union*(self.ink(image)<.5).float()
                self.restore=torch.maximum(self.restore,F.max_pool2d(missing,3,1,1))
            else:self.phase='unresolved'
        if self.phase!='erase':self.erase.zero_();self.update_support()
        elif index%2==0:self.refresh_erase(image)
        return dict(step=index,phase=self.phase,cycle=self.cycles,coverage_before=cov)

    def finish_step(self, image, row):
        cov=self.coverage(image);verified=min(cov)>=self.threshold
        if self.phase=='restore' and verified:self.phase='settle';self.settle_left=self.settle_steps
        elif self.phase=='settle':
            self.settle_left-=1
            if self.settle_left<=0 and verified:
                self.phase='erase';self.refresh_erase(image)
        row.update(coverage_after=cov,restoration_verified=verified,next_phase=self.phase,
                   erase_pixels=int((self.erase>.01).sum()))
        self.history.append(row)

    def __call__(self, image):
        loss,details=super().__call__(image)
        deficits=[CandidateObjective.mean_region(F.relu(.6-self.ink(image)),p) for p in self.required_paths]
        path_loss=torch.stack(deficits).mean()
        details['path_deficit']=float(path_loss.detach())
        return loss+path_loss,details

    def diagnostics(self, image):
        cov=self.coverage(image)
        keep=CandidateObjective.mean_region((image-self.source).square(),self.fixed_preserve)
        return dict(per_stroke_coverage=cov,min_coverage=min(cov),normal_region_mse=float(keep),
                    all_paths_present=min(cov)>=self.threshold)
