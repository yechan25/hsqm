import unittest
import torch
from stroke_model.local_reference_edit import LocalReferenceEdit,optimize_local

class LocalEditTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def sample(self):
        ref=torch.ones(1,3,32,32);ref[:,:,4:28,16:19]=0;ref[:,:,14:17,19:26]=0
        bad=ref.clone();bad[:,:,14:17,9:16]=0
        return ref,bad
    def test_erase_spur_preserve_vertical(self):
        ref,bad=self.sample();e=LocalReferenceEdit(ref,bad)
        self.assertEqual(float(e.erase[:,:,14:17,9:15].min()),1)
        self.assertEqual(float(e.protect[:,:,4:28,16:19].min()),1)
        result,trace=optimize_local(bad,lambda z:z,e,iterations=40,budget_ratio=.4)
        self.assertTrue(trace['accepted'])
        self.assertGreater(float(result[:,:,14:17,9:15].mean()),.5)
        self.assertTrue(torch.equal(result[:,:,4:28,16:19],bad[:,:,4:28,16:19]))
    def test_clean_reference_no_edit(self):
        ref,_=self.sample();e=LocalReferenceEdit(ref,ref)
        out,t=optimize_local(ref,lambda z:z,e)
        self.assertFalse(t['accepted']);self.assertTrue(torch.equal(out,ref))
    def test_coupled_decoder_rejects_damage(self):
        ref,bad=self.sample();e=LocalReferenceEdit(ref,bad)
        z=torch.ones(1)
        out,t=optimize_local(z,lambda v:bad+(v-1),e,iterations=10,
            max_protected_rms=0,max_protected_peak=0)
        self.assertFalse(t['accepted']);self.assertTrue(torch.equal(out,z))
    def test_backtracking_recovers_feasible_smaller_step(self):
        ref,bad=self.sample();e=LocalReferenceEdit(ref,bad)
        z=torch.ones(1)
        out,t=optimize_local(z,lambda v:bad+(v-1),e,iterations=3,
            step_ratio=.1,budget_ratio=.5,protect_weight=.01,
            max_protected_rms=.02,max_protected_peak=.03)
        self.assertTrue(t['accepted'])
        self.assertTrue(any(r['accepted'] and r['backtrack']>0 for r in t['local_iterations']))
        m=e.measure(bad+(out-1))
        self.assertLessEqual(float(m['protected'])**.5,.02)
        self.assertLess(float(m['edit']),float(e.measure(bad)['edit']))

    def test_halo_cleans_faint_boundary_without_erasing_reference(self):
        ref,bad=self.sample();bad[:,:,13:14,9:15]=.9
        old=LocalReferenceEdit(ref,bad)
        new=LocalReferenceEdit(ref,bad,halo_radius=2)
        self.assertEqual(float(old.erase[:,:,13:14,9:15].sum()),0)
        self.assertGreater(float(new.erase[:,:,13:14,9:15].sum()),0)
        self.assertEqual(float(new.erase[:,:,4:28,16:19].sum()),0)
        result,trace=optimize_local(bad,lambda z:z,new,iterations=48,budget_ratio=.4)
        self.assertTrue(trace['accepted'])
        self.assertGreater(float(result[:,:,13:14,9:15].mean()),.9)

    def test_faint_residue_retains_erase_gradient(self):
        ref,bad=self.sample()
        mse=LocalReferenceEdit(ref,bad)
        hybrid=LocalReferenceEdit(ref,bad,erase_l1_weight=.5)
        faint=ref.clone();faint[:,:,14:17,9:15]=.98
        x=faint.requires_grad_(True)
        gm,=torch.autograd.grad(mse.measure(x)['edit'],x)
        gh,=torch.autograd.grad(hybrid.measure(x)['edit'],x)
        self.assertGreater(float(gh.abs().sum()),float(gm.abs().sum())*2)
        self.assertLess(float(gh[:,:,14:17,9:15].sum()),0)
        hybrid.rebase(faint.detach())
        out,t=optimize_local(faint.detach(),lambda z:z,hybrid,iterations=48)
        self.assertLess(t['residual_mean_after'],t['residual_mean_before'])
        self.assertTrue(torch.equal(out[:,:,4:28,16:19],ref[:,:,4:28,16:19]))
        self.assertIn(t['stop_reason'],('iteration_limit','zero_gradient','protection_limit','no_feasible_descent'))

    def test_tail_focuses_gradient_on_remaining_dark_patch(self):
        ref,bad=self.sample()
        plain=LocalReferenceEdit(ref,bad,erase_l1_weight=.5)
        focused=LocalReferenceEdit(ref,bad,erase_l1_weight=.5,erase_tail_weight=.5)
        current=ref.clone();current[:,:,14:17,9:16]=.98;current[:,:,15:16,10:11]=.8
        x=current.requires_grad_(True)
        g0,=torch.autograd.grad(plain.measure(x)['edit'],x)
        g1,=torch.autograd.grad(focused.measure(x)['edit'],x)
        self.assertGreater(float(g1[:,:,15:16,10:11].abs().sum()),float(g0[:,:,15:16,10:11].abs().sum())*2)
        focused.rebase(current.detach())
        out,t=optimize_local(current.detach(),lambda z:z,focused,iterations=48)
        self.assertLess(t['erase_tail_after'],t['erase_tail_before'])
        self.assertTrue(torch.equal(out[:,:,4:28,16:19],ref[:,:,4:28,16:19]))

    def test_maps_stay_fixed_after_rebase(self):
        ref,bad=self.sample();e=LocalReferenceEdit(ref,bad);mask=e.erase.clone()
        e.rebase(ref);self.assertTrue(torch.equal(mask,e.erase))

if __name__=='__main__':unittest.main()
