# Topo: local direction and stroke relations

The V2 network, softmax partition, flow, head, inference, and state dict are
unchanged. This revision adds an optional **training loss**, not trained weights.
The score model is unchanged. Existing Topo tests and canvas accept new weights.

## Research basis and exact scope

[Adaptive Affinity Fields, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Jyh-Jing_Hwang_Adaptive_Affinity_Field_ECCV_2018_paper.html)
supervises whether nearby semantic labels belong to the same or different region,
using KL-based attraction and margin separation. It reports improvements on
Cityscapes, PASCAL VOC, and GTA5, without inference-time parameters.

We adapt that relationship supervision to sparse stroke owners. This is **not**
the complete AAF algorithm: no adversarial radius learning or published AAF
performance claim. Fixed offsets along four axes at radii 1, 3, 5 limit work.
Symmetric KL attracts same-owner distributions; relu(1 - KL) separates different
owners. Relations are balanced separately by endpoint owner and relation type.
The existing weak-owner loss only had same-owner nearest-neighbor consistency;
this adds explicit different-owner separation and longer neighborhood relations.

Local PCA over a 7x7 window of actual input annotation seeds estimates an axial
tangent. Tangential same-owner and transverse different-owner pairs receive
weights between 1 and 2. Ambiguous corners/isotropic neighborhoods have reduced
directional emphasis. Tangents are local, not a global straight-line constraint:
curves are allowed. There is no explicit straight/corner/loop classifier or
closed-loop guarantee. Reference conditioning still comes from the original V2
head; this loss learns from actual input owners rather than treating an imperfect
reference-to-input warp as truth. Missing/absent strokes get no fabricated seeds.

Only pairs with two known owners contribute. Unknown/background pixels are never
negative labels. If conservative cached owners are already enabled, their
reliability also gates pairs; tangent estimation still uses original seeds.
There is no matching against thin masks as dense foreground targets, no skeleton
loss added here, and no stroke-width target. Existing CE/Dice and seed focal loss
remain essential: pair losses alone cannot recover missing strokes, and KL at
identical distributions has zero first derivative. Do not train with pair loss
alone. This is a proposed adaptation, not a guarantee against contamination.

## Execution and validation

`notebooks/train_topology_zips.ipynb` defaults to weight **0.03**, every **2**
batches. Trainer defaults remain disabled for backwards compatibility. Set
`DIRECTION_AFFINITY_WEIGHT=0` for the control run. Radius/window/margin are fixed
implementation constants in this revision; checkpoints store weight and interval,
and notebook saves the git commit. No new datasets, caches or model branches.

For a pilot, set `INIT_CHECKPOINT` to the current After `best.pt`, `EPOCHS=3`,
`LEARNING_RATE=1e-4`. Use identical initializer, seed, and other settings for the
weight-0 control. Both runs create new folders. This is initialization with a new
optimizer, not optimizer/epoch resume. Compare through the existing comparison
notebook and the same saved canvas drawings of 초/아 plus curved/bent strokes.
Inspect contamination, missing small strokes, and original width; annotation Dice
alone may miss contamination in unlabeled ink. Best checkpoint selection still
uses existing validation macro Dice, not a newly validated shape metric.

History records directional loss and application count. Check `epoch_seconds`
against the 30s baseline on the same GPU; notebook warning budget remains 33s.
The new term has no inference cost but GPU training overhead is **not measured**
here. If over budget, use interval 4 in both experiments or disable the term;
do not claim a speed guarantee from CPU unit tests.

Synthetic tests check vertical/horizontal/local isotropic direction, a stolen
horizontal arm, corrective gradients, unknown-pixel exclusion, rotations, channel
permutations, absent padding, and reliability gating. These establish operator
behavior, not improvement on the user's trained model.
