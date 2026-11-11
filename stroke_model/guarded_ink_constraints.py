"""Conservative release policy: static base, consensus updates, stroke guards.

No quality guarantees: rejected edits/fallbacks are explicitly reported. All arms
remain available; selected.png is copied from one complete output, never composited.
"""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import distance_transform_edt

from diffusion_model.fitted_candidate_diffusion import InkConstraintObjective
from .adaptive_ink_constraints import AdaptiveInkConstraints
from .stroke_reference_fit import principal_path


class GuardedInkConstraints(AdaptiveInkConstraints):
    def __init__(self, source, candidate, extractor, records, *, enable_restore=False,
                 confirmations=2, refresh_every=3, mse_slack=.001,
                 local_slack=.025, ink_drop_slack=.02):
        if type(confirmations) is not int or confirmations < 2:
            raise ValueError('At least two confirmations required')
        if any(not np.isfinite(v) or v < 0 for v in (mse_slack, local_slack, ink_drop_slack)):
            raise ValueError('Invalid preservation limits')
        super().__init__(source, extractor, records, enable_restore=enable_restore, refresh_every=refresh_every)
        self.confirmations = confirmations
        self.slacks = source.new_tensor([mse_slack, local_slack, ink_drop_slack])
        # The successful sparse fixed-corridor method is the default foundation.
        base = InkConstraintObjective(source, candidate, enable_restore=enable_restore)
        self.base_erase, self.base_restore = base.erase.clone(), base.restore.clone()
        self.branch_masks = []
        self.uncertain = torch.zeros_like(self.base_erase)
        source_ink = self.ink(source)
        for r in self.records:
            observed = self.crop(source_ink, r['box'], r['size'])[0, 0].cpu().numpy()
            masks = []
            for k in range(len(r['probabilities'])):
                proposal = self.straight_branch(r, k, observed)
                masks.append(proposal)
                canvas = torch.zeros_like(self.uncertain)
                self.place(canvas, self.tensor(proposal), r['box'])
                self.uncertain = torch.maximum(self.uncertain, canvas)
            self.branch_masks.append(np.stack(masks))
        self.pending = torch.zeros_like(self.base_erase, dtype=torch.int32)
        self.dynamic = torch.zeros_like(self.base_erase)
        excluded = F.max_pool2d(((self.base_erase > .01) | (self.base_restore > .01) |
                                 (self.uncertain > .05)).float(), 7, 1, 3)
        self.healthy_groups = []
        for r in self.records:
            for p in r['probabilities']:
                canvas = torch.zeros_like(source_ink)
                self.place(canvas, self.tensor(p), r['box'])
                group = canvas*(source_ink > .2).float()*(1-excluded)
                if float(group.sum()) > 1:self.healthy_groups.append(group)
        self.groups = torch.cat(self.healthy_groups, 0) if self.healthy_groups else torch.zeros_like(source_ink)
        self.healthy_union = self.groups.sum(0, keepdim=True).clamp(0, 1)
        self.baseline, self.arm_states = {}, {}
        self.current_step = None
        self.erase, self.restore = self.base_erase.clone(), self.base_restore.clone()
        self.update_support()
        self.fixed = dict(erase=self.erase.clone(), restore=self.restore.clone(), preserve=self.preserve.clone())
        self.fixed_error = torch.maximum(self.erase, self.restore).clone()
        self.selection_erase = torch.maximum(self.base_erase, self.uncertain).clone()
        self.history = [dict(step=-1, erase_pixels=int((self.erase > .01).sum()),
                             restore_pixels=int((self.restore > .01).sum()), policy='static_base')]
        self.metadata.update(kind='guarded_ink_release', confirmations=confirmations,
            fixed_base=True, dynamic_masks_revocable=True, normal_groups_frozen=True,
            mse_slack=mse_slack, local_slack=local_slack, ink_drop_slack=ink_drop_slack,
            original_prototype_rgb_target=False, straight_branch_proposal_pixels=int((self.uncertain > .05).sum()))
        self.ready = True

    def __call__(self, image):
        loss, details = super().__call__(image)
        if not getattr(self, 'ready', False):return loss, details
        # Counter VAE leakage before a hard guard rejects the proposal. Region-
        # normalized stroke errors are not diluted by a large white background.
        error = (image-self.source).square().mean(1, keepdim=True)
        mass = self.groups.sum((1, 2, 3)).clamp_min(1)
        stroke_error = (error*self.groups).sum((1, 2, 3))/mass
        pooled_mass = F.avg_pool2d(self.groups, 5, 1, 2)
        local = F.avg_pool2d(error*self.groups, 5, 1, 2)/pooled_mass.clamp_min(.1)
        boundary = local.flatten(1).amax(1).mean()
        details['normal_stroke_mse'] = float(stroke_error.mean().detach())
        details['normal_local_mse'] = float(boundary.detach())
        return loss+stroke_error.mean()+boundary, details

    def straight_branch(self, r, k, observed):
        """Propose side bulges of a straight prototype; do not target its caps."""
        empty = np.zeros_like(observed, dtype=np.float32)
        try:
            canonical, _ = principal_path(r['strokes'][0, k].cpu().numpy() > .2)
            chord = np.linalg.norm(canonical[-1]-canonical[0])
            length = np.linalg.norm(np.diff(canonical, axis=0), axis=1).sum()
            if chord/max(length, 1e-6) < .98:return empty
            path, _ = principal_path(r['skeletons'][k])
        except ValueError:
            return empty
        a, b = path[0], path[-1]
        axis = b-a;length = float(np.linalg.norm(axis))
        if length < 5:return empty
        axis /= length;normal = np.array([-axis[1], axis[0]])
        yy, xx = np.indices(observed.shape)
        xy = np.stack((xx, yy), -1)-a
        along, across = xy@axis, xy@normal
        body = (observed > .35) & (r['probabilities'][k] > .5)
        widths = distance_transform_edt(body)[r['skeletons'][k] & body & (along > .2*length) & (along < .8*length)]
        if widths.size < 4:return empty
        calibrated = float(np.median(r['radii'][k][r['skeletons'][k]]))
        radius = float(np.clip(np.quantile(widths, .25), .65*calibrated, calibrated))
        # Restrict to the middle; irregular stroke caps are deliberately untouched.
        excess = np.clip((np.abs(across)-radius-.6)/max(radius*.4, .5), 0, 1)
        return (excess*(along > .1*length)*(along < .9*length)*body).astype(np.float32)

    def update_support(self):
        self.edit = torch.maximum(self.erase, self.restore)
        self.preserve = 1-F.max_pool2d((self.edit > .01).float(), 5, 1, 2)

    @torch.no_grad()
    def refresh(self, image, *, step, initial=False):
        if not getattr(self, 'ready', False):return super().refresh(image, step=step, initial=initial)
        old_erase, old_restore = self.erase, self.restore
        self.erase = torch.zeros_like(old_erase);self.restore = torch.zeros_like(old_restore)
        row = super().refresh(image, step=step)
        raw = self.erase.clone()
        self.erase, self.restore = old_erase, old_restore
        # Branch proposals inside a broad geometric corridor need separate evidence.
        current_ink = self.ink(image).clamp(0, 1)
        branches = torch.zeros_like(raw)
        for r, masks in zip(self.records, self.branch_masks):
            crop = self.crop(current_ink, r['box'], r['size'])[0, 0].cpu().numpy()
            weight = (masks*r['probabilities']).sum(0)*(crop > .12)
            self.place(branches, self.tensor(weight), r['box'])
        raw = torch.maximum(raw, branches)
        # Existing healthy ink is not reclassified merely to make a new mask fit.
        raw = raw*(self.healthy_union < .1).float()
        positive = raw > .05
        self.pending = torch.where(positive, self.pending+1, torch.zeros_like(self.pending))
        confirmed = self.pending >= self.confirmations
        # New masks need repeated evidence. Lost evidence can revoke old additions.
        self.dynamic = torch.where(confirmed, raw, self.dynamic*.6)
        self.dynamic[self.dynamic < .02] = 0
        self.erase = torch.maximum(self.base_erase, self.dynamic)
        self.restore = self.base_restore.clone()
        self.erase = self.erase*(self.restore <= 0).float()
        self.update_support()
        row.update(erase_pixels=int((self.erase > .01).sum()),
                   confirmed_pixels=int(confirmed.sum()), pending_pixels=int((self.pending == 1).sum()))
        return row

    @torch.no_grad()
    def guard_state(self, image):
        rgb = image.clamp(0, 1)
        error = (rgb-self.source).square().mean(1, keepdim=True)
        weights = self.groups
        mass = weights.sum((1, 2, 3)).clamp_min(1)
        mse = (error*weights).sum((1, 2, 3))/mass
        pooled_mass = F.avg_pool2d(weights, 5, 1, 2)
        local = F.avg_pool2d(error*weights, 5, 1, 2)/pooled_mass.clamp_min(.1)
        peak = local.flatten(1).amax(1)
        drop = (F.relu(self.ink(self.source)-self.ink(rgb))*weights).sum((1, 2, 3))/mass
        return torch.stack((mse, peak, drop), -1)

    def record_baseline(self, step, image):self.baseline[step] = self.guard_state(image)
    def begin_step(self, step):self.current_step = step

    def check_proposal(self, before, after):
        if self.current_step not in self.baseline:raise RuntimeError('Matched baseline guard missing')
        # An already drifting anchor may improve but must not get any worse.
        bound = torch.maximum(self.baseline[self.current_step]+self.slacks, self.guard_state(before))
        return bool((self.guard_state(after) <= bound+1e-7).all())

    def finish_arm(self, name, image):
        error = self.error_value(image)
        self.arm_states[name] = dict(guard=self.guard_state(image), error=error)

    def error_value(self, image):
        ink = self.ink(image)
        erase = ((ink-self.background_ink).clamp_min(0)*self.selection_erase).sum()/self.selection_erase.sum().clamp_min(1)
        restore = ((self.required_ink-ink).clamp_min(0)*self.base_restore).sum()/self.base_restore.sum().clamp_min(1)
        return float(erase+restore)

    def select_result(self, guided_name):
        base, guided = self.arm_states['redenoise_only'], self.arm_states[guided_name]
        normal_ok = bool((guided['guard'] <= base['guard']+self.slacks+1e-7).all())
        initial = self.error_value(self.source)
        improved = guided['error'] < min(base['error'], initial)-1e-5
        accepted = normal_ok and improved
        return dict(file=guided_name+'.png' if accepted else 'input.png',
            status='guided_passed_internal_checks' if accepted else 'original_retained',
            normal_guard_passed=normal_ok, fixed_error_improved=improved,
            original_error=initial, baseline_error=base['error'], guided_error=guided['error'],
            guard_worst_guided=guided['guard'].amax(0).cpu().tolist(),
            guard_worst_baseline=base['guard'].amax(0).cpu().tolist(),
            note='Internal heuristic checks, not verified visual correctness; all raw arms retained')
