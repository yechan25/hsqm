import unittest
import torch
from stroke_model.directional_affinity import seed_directions,directional_affinity_loss


class DirectionalAffinityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def fixture(self):
        labels=torch.full((1,24,24),-1,dtype=torch.long)
        labels[0,3:18,12]=0
        labels[0,18,3:22]=1
        logits=torch.zeros(1,2,24,24)
        logits[:,0]=2;logits[:,1]=-2
        logits[0,0,18,:]=-2;logits[0,1,18,:]=2
        return labels,logits

    def test_local_axial_directions_and_ambiguous_corner(self):
        labels,_=self.fixture();c,s,q=seed_directions(labels,2)
        self.assertLess(float(c[0,10,12]),-.99)
        self.assertGreater(float(c[0,18,6]),.99)
        self.assertLess(float(s[0,10,12].abs()),.01)
        self.assertGreater(float(q[0,10,12]),.99)
        self.assertEqual(float(q[0,0,0]),0)
        # Isotropic local loop must not be assigned a confident straight axis.
        circle=torch.full((1,17,17),-1,dtype=torch.long)
        circle[0,8,8]=0
        for y,x in ((6,8),(10,8),(8,6),(8,10)):circle[0,y,x]=0
        self.assertLess(float(seed_directions(circle,1)[2][0,8,8]),.01)

    def test_correct_partition_beats_bent_spill_and_gradient_repairs_it(self):
        labels,good=self.fixture();bad=good.clone()
        # Vertical owner steals the horizontal arm next to its endpoint.
        bad[0,0,18,12:18]=1.;bad[0,1,18,12:18]=-1.
        bad.requires_grad_()
        correct=directional_affinity_loss({'probabilities':good.softmax(1)},labels)
        error=directional_affinity_loss({'probabilities':bad.softmax(1)},labels)
        self.assertGreater(float(error),float(correct)+.1)
        error.backward()
        self.assertTrue(torch.isfinite(bad.grad).all())
        # Gradient descent lowers the stealing channel relative to the owner.
        self.assertGreater(float(bad.grad[0,0,18,14]),float(bad.grad[0,1,18,14]))

    def test_unknown_ink_is_not_background_or_forced_thin(self):
        labels,logits=self.fixture()
        first=directional_affinity_loss({'probabilities':logits.softmax(1)},labels)
        changed=logits.clone();changed[:,:,0:2,:]=torch.randn_like(changed[:,:,0:2,:])*5
        second=directional_affinity_loss({'probabilities':changed.softmax(1)},labels)
        self.assertAlmostEqual(float(first),float(second),places=6)

    def test_channel_permutation_and_rotation_are_invariant(self):
        labels,logits=self.fixture();p=logits.softmax(1)
        base=directional_affinity_loss({'probabilities':p},labels)
        swapped=torch.where(labels>=0,1-labels,labels)
        other=directional_affinity_loss({'probabilities':p.flip(1)},swapped)
        rotated=directional_affinity_loss({'probabilities':p.rot90(1,(-2,-1))},labels.rot90(1,(-2,-1)))
        self.assertAlmostEqual(float(base),float(other),places=5)
        self.assertAlmostEqual(float(base),float(rotated),places=5)

    def test_absent_padding_and_all_unknown_have_finite_zero_gradient(self):
        labels,logits=self.fixture();p=logits.softmax(1)
        padded=torch.cat((p,torch.zeros_like(p[:,:1])),1)
        self.assertAlmostEqual(float(directional_affinity_loss({'probabilities':p},labels)),
            float(directional_affinity_loss({'probabilities':padded},labels)),places=5)
        unknown=torch.full_like(labels,-1);logits.requires_grad_()
        loss=directional_affinity_loss({'probabilities':logits.softmax(1)},unknown)
        loss.backward();self.assertEqual(float(loss),0)
        self.assertEqual(float(logits.grad.abs().sum()),0)

    def test_cached_owner_pairs_reject_untrusted_pixels(self):
        labels,logits=self.fixture();owners=labels.clone();owners[0,:2]=1
        reliability=(labels>=0).float()
        base=directional_affinity_loss({'probabilities':logits.softmax(1)},labels)
        cached=directional_affinity_loss({'probabilities':logits.softmax(1)},labels,owners,reliability)
        self.assertAlmostEqual(float(base),float(cached),places=6)


if __name__=='__main__':unittest.main()
