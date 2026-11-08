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
    def test_maps_stay_fixed_after_rebase(self):
        ref,bad=self.sample();e=LocalReferenceEdit(ref,bad);mask=e.erase.clone()
        e.rebase(ref);self.assertTrue(torch.equal(mask,e.erase))

if __name__=='__main__':unittest.main()
