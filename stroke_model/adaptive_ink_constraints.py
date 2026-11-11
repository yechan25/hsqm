"""Frozen stroke identities, per-stroke bands, periodically refreshed ownership.

Geometry is anchored to the initial fitted strokes, not re-fitted to new errors.
Extractor softmax is detached; only image ink losses backpropagate through VAE.
"""
import hashlib
import io
import json
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt, binary_dilation
from skimage.morphology import skeletonize

from diffusion_model.fitted_candidate_diffusion import InkConstraintObjective, CandidateObjective
from .stroke_reference_fit import _component, principal_path


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_stroke_bundle(archive, case_id):
    """Only known ZIP members are read, no recorded absolute paths are followed."""
    with zipfile.ZipFile(archive) as z:
        info = json.loads(z.read('experiment_info.json'))
        manifest = json.loads(z.read('source_conditions/conditions.json'))
        records = []
        if len(manifest) != 4:
            raise ValueError('Exactly four reference slots required')
        for i, entry in enumerate(manifest):
            name = entry['path'].replace('\\', '/').rsplit('/', 1)[-1]
            with np.load(io.BytesIO(z.read('source_conditions/'+name)), allow_pickle=False) as d:
                reference, strokes = d['reference'].copy(), d['strokes'].copy()
            with np.load(io.BytesIO(z.read(f'{case_id}/warped_slot_{i+1}.npz')), allow_pickle=False) as d:
                fitted = d['warped_strokes'].copy()
            if strokes.shape != fitted.shape or reference.shape != (1, *strokes.shape[-2:]):
                raise ValueError('Reference/fitted stroke shape mismatch')
            records.append(dict(reference=reference, strokes=strokes, fitted=fitted))
        return records, info['checkpoint_sha256']


class AdaptiveInkConstraints(InkConstraintObjective):
    def __init__(self, source, extractor, records, *, refresh_every=3, enable_restore=False,
                 margin_ratio=.2, margin_min=.6, margin_max=1.8,
                 ownership_blend=.35, mask_blend=.5, preserve_weight=.5, continuity_weight=.2):
        if source.shape[:2] != (1, 3) or len(records) != 4:
            raise ValueError('One RGB sheet and four stroke records required')
        if type(refresh_every) is not int or refresh_every < 1 or not 0 < ownership_blend <= 1 or not 0 < mask_blend <= 1:
            raise ValueError('Invalid refresh schedule/blending')
        if not 0 <= margin_ratio or not 0 < margin_min <= margin_max or preserve_weight < 0 or continuity_weight < 0:
            raise ValueError('Invalid stroke bands/weights')
        self.source = source.detach()
        self.extractor = extractor.eval().requires_grad_(False)
        self.refresh_every, self.ownership_blend, self.mask_blend = refresh_every, ownership_blend, mask_blend
        self.enable_restore = enable_restore
        self.preserve_weight, self.continuity_weight = preserve_weight, continuity_weight
        self.background_ink, self.required_ink = .015, .7
        self.records, self.bands, self.history = [], [], []
        h, w = source.shape[-2:]
        if h % 2 or w % 2:
            raise ValueError('Equal 2x2 slots require even sheet dimensions')
        boxes = [(0, 0, w//2, h//2), (w//2, 0, w, h//2),
                 (0, h//2, w//2, h), (w//2, h//2, w, h)]
        source_ink = self.ink(source)
        # Identity k always refers to input canonical stroke k, never sorted/merged.
        for slot, (raw, box) in enumerate(zip(records, boxes)):
            ref = torch.as_tensor(raw['reference'], device=source.device, dtype=torch.float32)[None]
            strokes = torch.as_tensor(raw['strokes'], device=source.device, dtype=torch.float32)[None]
            fitted = np.asarray(raw['fitted'], np.float32)
            crop = self.crop(source_ink, box, fitted.shape[-2:])
            probabilities = self.predict(crop, ref, strokes)
            observed = crop[0, 0].cpu().numpy()
            distances, radii, margins, cores, source_bodies = [], [], [], [], []
            for k, mask in enumerate(fitted):
                binary = mask > .35
                skeleton = skeletonize(binary)
                if not skeleton.any():
                    raise ValueError(f'Slot {slot+1}, stroke {k+1}: empty fitted path')
                inside = distance_transform_edt(binary)
                distance, nearest = distance_transform_edt(~skeleton, return_indices=True)
                expected = max(float(np.median(inside[skeleton])), .75)
                body = (observed > .35) & (probabilities[k] > .35)
                width_samples = distance_transform_edt(body)[skeleton & body]
                # Only samples on the anchored main path calibrate width. Extras
                # off the path cannot enlarge the allowed corridor arbitrarily.
                observed_half = float(np.median(width_samples)) if width_samples.size >= 4 else expected
                half = float(np.clip(observed_half, .75*expected, 1.25*expected))
                local_radius = np.maximum(inside[tuple(nearest)], .75)*(half/expected)
                margin = float(np.clip(margin_ratio*2*half, margin_min, margin_max))
                trusted_core = skeleton & (observed > .35)
                source_body = np.zeros_like(binary)
                path_protected = False
                try:
                    component, _ = _component(probabilities[k]*observed, hint=mask)
                    path, path_info = principal_path(component, n=128)
                    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                    ratio = length/max(int(skeleton.sum()), 1)
                    if not path_info['closed'] and .65 <= ratio <= 1.5 and path_info['path_coverage'] > .65:
                        dense = np.stack([np.interp(np.linspace(0, 1, 512), np.linspace(0, 1, len(path)), path[:, j]) for j in range(2)], 1)
                        xy = np.rint(dense).astype(int)
                        xy[:, 0] = xy[:, 0].clip(0, binary.shape[1]-1)
                        xy[:, 1] = xy[:, 1].clip(0, binary.shape[0]-1)
                        source_path = np.zeros_like(binary);source_path[xy[:, 1], xy[:, 0]] = True
                        if float(np.median(distance[source_path])) <= expected+margin:
                            # Protect the original main body, not the smooth
                            # candidate outline. Cap radius to exclude side branches.
                            source_body = (distance_transform_edt(~source_path) <= 1.1*half) & (observed > .35)
                            trusted_core = source_path & (observed > .35)
                            path_protected = True
                except ValueError:
                    pass  # Explicitly reported fallback to anchored-path protection.
                distances.append(distance); radii.append(local_radius); margins.append(margin); cores.append(trusted_core)
                source_bodies.append(source_body)
                self.bands.append(dict(slot=slot+1, stroke=k+1, half_width_model=half,
                    tolerance_model=margin, tolerance_sheet_x=margin*(box[2]-box[0])/fitted.shape[-1],
                    tolerance_sheet_y=margin*(box[3]-box[1])/fitted.shape[-2],
                    trusted_core_pixels=int(trusted_core.sum()), original_main_path_protected=path_protected))
            self.records.append(dict(reference=ref, strokes=strokes, box=box,
                size=fitted.shape[-2:], probabilities=probabilities,
                distances=np.stack(distances), radii=np.stack(radii), margins=np.asarray(margins)[:, None, None],
                cores=np.stack(cores), original_bodies=np.stack(source_bodies),
                skeletons=np.stack([skeletonize(s > .35) for s in fitted])))
        self.erase = torch.zeros_like(source_ink)
        self.restore = torch.zeros_like(source_ink)
        self.protected_cores = []
        for record in self.records:
            for core in record['cores']:
                canvas = torch.zeros_like(source_ink)
                self.place(canvas, self.tensor(core.astype(np.float32)), record['box'])
                self.protected_cores.append(canvas)
        self.refresh(source, step=-1, initial=True)
        self.fixed = dict(erase=self.erase.clone(), restore=self.restore.clone(), preserve=self.preserve.clone())
        self.metadata = dict(kind='adaptive_stroke_ink_constraints', candidate_rgb_matching=False,
            enable_restore=enable_restore, refresh_every=refresh_every, stroke_identities_frozen=True,
            geometry_frozen=True, width_calibration='robust trusted main-path samples, frozen',
            ownership_backward=False, bands=self.bands, preserve_weight=preserve_weight,
            continuity_weight=continuity_weight, masks_frozen_for_entire_pass=False)

    def tensor(self, a):
        return torch.as_tensor(np.asarray(a).copy(), dtype=torch.float32, device=self.source.device)[None, None]

    @staticmethod
    def crop(image, box, size):
        x0, y0, x1, y1 = box
        return F.interpolate(image[..., y0:y1, x0:x1], size=size, mode='bilinear', align_corners=False)

    @staticmethod
    def place(canvas, patch, box):
        x0, y0, x1, y1 = box
        canvas[..., y0:y1, x0:x1] = F.interpolate(patch, size=(y1-y0, x1-x0), mode='bilinear', align_corners=False)

    @torch.no_grad()
    def predict(self, ink, reference, strokes):
        p = self.extractor(ink, reference, strokes)['probabilities'][0].float()
        if p.shape != strokes.shape[1:] or not torch.isfinite(p).all() or (p < 0).any():
            raise ValueError('Invalid extractor ownership probabilities')
        p = p/p.sum(0, keepdim=True).clamp_min(1e-8)
        return p.cpu().numpy()

    @torch.no_grad()
    def refresh(self, image, *, step, initial=False):
        ink = self.ink(image).clamp(0, 1)
        erase, restore = torch.zeros_like(ink), torch.zeros_like(ink)
        rows = []
        for slot, r in enumerate(self.records):
            crop = self.crop(ink, r['box'], r['size'])
            current = self.predict(crop, r['reference'], r['strokes'])
            p = current if initial else (1-self.ownership_blend)*r['probabilities']+self.ownership_blend*current
            r['probabilities'] = p
            observed = crop[0, 0].cpu().numpy()
            excess = r['distances']-r['radii']
            outside = np.clip((excess-r['margins'])/np.maximum(r['margins'], .1), 0, 1)
            # Protect every other legitimate stroke at intersections too.
            safe_body = (excess <= r['margins']).any(0) | r['original_bodies'].any(0)
            responsibility = (p*outside).sum(0)
            # Existing faint residuals keep their old mask. New regions require
            # more than faint decoder noise before they can expand deletion.
            seeds = (observed > (.25 if initial else .12)) & (responsibility > .4) & ~safe_body
            grown = binary_dilation(seeds, iterations=1)
            weight = grown*responsibility*(~safe_body)
            missing = r['skeletons'].any(0) & (distance_transform_edt(~(observed > .2)) > 1.)
            restore_weight = missing.astype(np.float32) if self.enable_restore else np.zeros_like(weight)
            weight[restore_weight > 0] = 0
            self.place(erase, self.tensor(weight), r['box'])
            self.place(restore, self.tensor(restore_weight), r['box'])
            for k in range(len(p)):
                rows.append(dict(slot=slot+1, stroke=k+1,
                    assigned_ink=float((p[k]*observed).sum()),
                    outside_ink=float((p[k]*observed*outside[k]*(~safe_body)).sum())))
        # Retain previous deletion support: fading errors must not fall out of
        # the mask or become protected just because they were partly erased.
        self.erase = erase if initial else torch.maximum(self.erase, self.mask_blend*erase)
        self.restore = restore if initial else (1-self.mask_blend)*self.restore+self.mask_blend*restore
        self.erase = self.erase*(self.restore <= 0).float()
        self.edit = torch.maximum(self.erase, self.restore)
        self.preserve = 1-F.max_pool2d((self.edit > .01).float(), 5, 1, 2)
        row = dict(step=step, erase_pixels=int((self.erase > .01).sum()),
                   restore_pixels=int((self.restore > .01).sum()), per_stroke=rows)
        self.history.append(row)
        return row

    def maybe_refresh(self, image, step):
        return self.refresh(image, step=step) if step % self.refresh_every == 0 else None

    def export_state(self, path):
        arrays = {f'slot_{i+1}_probabilities': r['probabilities'] for i, r in enumerate(self.records)}
        arrays.update(erase=self.erase.detach().cpu().numpy(), restore=self.restore.detach().cpu().numpy())
        np.savez_compressed(path, **arrays)

    def __call__(self, image):
        loss, details = super().__call__(image)
        ink = self.ink(image)
        # Protect initially present main paths, averaged per stroke, not per sheet.
        continuity = [CandidateObjective.mean_region(F.relu(.65-ink), core)
                      for core in self.protected_cores if float(core.sum()) > 0]
        value = torch.stack(continuity).mean() if continuity else loss*0
        details['trusted_path_deficit'] = float(value.detach())
        return loss+self.continuity_weight*value, details

    def evaluate_fixed(self, image):
        current = dict(erase=self.erase, restore=self.restore, preserve=self.preserve)
        try:
            for key, value in self.fixed.items():
                setattr(self, key, value)
            return self(image)
        finally:
            for key, value in current.items():
                setattr(self, key, value)
