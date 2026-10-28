# Literature review and architecture decisions

## Review status and chronology

Before implementation began, the review consisted of abstracts, project pages, and
the one-page sprite-generation poster. It was **not a thorough reading of all ten
papers**. The dataset tools and compact supervised U-Net were started as an
engineering baseline, not as a reproduction or a literature-validated architecture.

Following the user's question, the downloaded PDFs were converted to text and the
sections below were read. This review covers methods, relevant training procedures,
and selected evaluations/limitations. It does not claim every appendix, figure,
equation, or implementation has been audited, nor that any published result has
been independently reproduced. Exact PDF hashes are in `papers/manifest.json`.

| Paper | Reading performed in this review |
| --- | --- |
| Sprite Sheet Diffusion | Main text §§1–8; dataset table in Appendix B; quantitative ablations in Appendix D.1. |
| Animate Anyone | Architecture/training in §3; implementation in §4.1; comparisons, ablations, and limitations in §§4–5. |
| LayerDiffuse | Introduction and methods §§1–3, including data/training; selected output, robustness, speed, and limitations discussion in §§4.8–4.12. |
| IP-Adapter | Method §3; setup §4.1; ablations §4.4; conclusions/limitations §5. |
| ControlNet | Method/training/inference §3; selected ablations, quantitative evaluation, and discussion in §4. |
| Qwen-Image-Layered | Method/training §3 and data, metrics, ablations, and reconstruction results in §4. |
| Text2Layer | Data construction/filtering in §III; model/objectives in §IV; implementation/results in §V; conclusions in §VI. |
| LayerDiff | Task/architecture/data in §4, excluding a detailed derivation of self-mask guidance; training/evaluation setup in §5; limitations; initialization/data details in Appendices A–C. |
| LaDe | RGBA VAE, prompt conditioning, and diffusion architecture in §3; implementation/data and selected results/limitations in §§4–5; training stages in §6.1. |
| Sprite Sheet Generation poster | Entire extracted one-page text, including method, dataset, and comparison. |

## Findings relevant to this project

### Sprite Sheet Diffusion

[Paper](https://arxiv.org/abs/2412.03685), §§2, 4–5 and Appendix D.1.

The task predicts full character images from a character reference and pose sequence.
It adapts ReferenceNet, a pose guider, and a temporal module with staged training.
The dataset contains 152 action sequences and separates unseen actions from unseen
characters. The authors report difficulty preserving fine details and props. Temporal
training can degrade identity/detail metrics; a reported failure loses a gun.
Stage-one training is reported to require over 30 GB GPU memory.

**Implication:** borrow reference/spatial conditioning ideas, but change the prediction
target to equipment-only RGBA. Explicitly test missing item parts. Do not assume a
temporal module improves our task, and do not extrapolate its GPU timings to Iris.
This paper does not demonstrate interchangeable equipment overlays.

### Animate Anyone

[Paper](https://arxiv.org/abs/2311.17117), §§3–5.

ReferenceNet supplies spatial appearance features through attention; a lightweight
pose guider supplies aligned structure. Stage one learns individual frames, and
stage two trains temporal layers while freezing the earlier modules. The paper
distinguishes appearance references, which are not spatially aligned with targets,
from aligned structural conditions. It reports hand instability and ambiguity when
new views reveal unseen surfaces.

**Implication:** an item illustration and a target hand/body region require different
conditioning paths. The existing spritesheet supplies structure, not a movement to
invent. Gloves need a dedicated failure set. Evidence for full-character video
generation does not establish equipment accuracy or RGBA support.

### LayerDiffuse

[Paper](https://arxiv.org/abs/2402.17113), §§3, 4.8–4.12.

Transparency is encoded as a regulated offset to an existing latent distribution,
with learned transparency encoding/decoding around a frozen RGB VAE. The diffusion
model is adapted to that representation. Multiple layers share attention, and
conditional variants can keep one layer clean while denoising another. The method
uses a substantial curated/synthetic transparency dataset, not merely an added
alpha output channel. RGB under zero alpha is padded for reconstruction/edge
behavior. Layer-specific illumination can reduce an asset's reuse.

**Implication:** a candidate RGBA representation and conditional-generation design,
not proof that pretrained transparency weights fit arbitrary equipment. Benchmark
encode/decode fidelity on tiny sprite parts before fine-tuning. Retain the original
body outside the generator and composite the result deterministically.

### IP-Adapter

[Paper](https://arxiv.org/abs/2308.06721), §§3, 4.1, 4.4, 5.

The adapter separates image and text cross-attention while freezing the pretrained
generator. Its basic reference representation uses global CLIP features; an ablation
examines finer grid features. The paper explicitly distinguishes resemblance in
content/style from highly consistent subject reproduction. Training the original
adapter used a large image-text dataset despite the adapter's modest parameter count.

**Implication:** evaluate pretrained reference conditioning rather than train a generic
adapter from our tiny dataset. Global semantic embeddings may discard glove seams,
emblems, or weapon geometry. Compare localized/spatial appearance features and
measure design fidelity, not just whether the output is recognizably a glove.

### ControlNet

[Paper](https://arxiv.org/abs/2302.05543), §§3–4.

ControlNet freezes a pretrained generator and adds a trainable copy of encoder/middle
blocks through zero-initialized connections. Its conditions are spatial maps such
as pose, edges, depth, and segmentation, trained with the diffusion objective. Those
maps guide generated structure; the architecture does not guarantee exact pixel
preservation or identity transfer from an unaligned reference.

**Implication:** a candidate for body/attachment guidance, not an equipment-appearance
solution on its own. Conditions must be derived from the supplied base or otherwise
available at inference. Supplying the ground-truth equipment mask during training
without an inference-time way to obtain it would invalidate the experiment.

### Qwen-Image-Layered

[Paper](https://arxiv.org/abs/2512.15603), §§3–4.

The method combines a shared RGB/RGBA VAE, variable-layer transformer, and staged
training from RGBA generation through multilayer generation to decomposition.
Training includes processed PSD layers and separate RGBA adaptation. Evaluation
uses alpha-weighted RGB error and alpha soft IoU, with layer alignment/merging to
handle decomposition ambiguity. Its reconstruction benchmark is not a low-resolution
equipment benchmark.

**Implication:** useful evidence for explicit RGBA representations and alpha-aware
metrics. Decomposing an existing picture does not fit a new item to multiple body
poses. Its full multilayer architecture is not automatically justified when our
required output is only one equipment overlay.

### Text2Layer

[Paper](https://arxiv.org/abs/2307.09781), §§III–VI.

The method constructs foreground/background/mask data using segmentation and
inpainting with quality filtering. A composition-aware autoencoder predicts image
components and masks before diffusion training. Its objectives include separate
mask terms and compositing-related supervision. Results show that improvements in
composite image quality do not necessarily improve mask quality.

**Implication:** evaluate alpha and appearance separately, and prefer existing true
equipment layers over automatically extracted masks when available. This paper
addresses layered image generation, not equipment motion. Its synthetic supervision
and training scale do not establish a recipe for our small pilot.

### LayerDiff

[Paper](https://arxiv.org/abs/2403.11929), §§4–5 and Appendices A–C.

LayerDiff models layer images and masks, with inter-layer attention and layer-specific
text conditioning. Its stated formulation uses nonoverlapping foreground masks and
binary assignment at sampling; it is not interchangeable with arbitrary overlapping
soft-alpha RGBA composition. Training data comes from detection/segmentation,
inpainting, and captioning. The authors identify data limitations for richer layers.

**Implication:** cross-layer coordination is informative, but the full formulation
does not directly match equipment with soft boundaries or transparency. Do not
confuse LayerDiff with LayerDiffuse, or treat either as a demonstrated animation
solution. Temporal relationships between item frames remain a separate question.

### LaDe

[Paper](https://arxiv.org/abs/2603.17965), §§3–6.1.

LaDe combines RGBA encoding, layer/role-aware positional encoding, and conditioning
on clean layers while denoising other layers. Its target is layered graphic media.
The paper reports an 11-billion-parameter diffusion model trained using 256 H100s,
large private datasets, and multiple stages. Limitations include memory consumption
and dependence on expanded prompts. Its evaluation is not an equipment-fitting test.

**Implication:** useful conditional-layer concepts, but training this full system is
not supported merely by having cluster access. Treat it as architectural context;
do not infer a deployable or reproducible equipment model from its published results.

### Sprite Sheet Generation poster

[Poster](https://marcovolino.github.io/docs/papers/2025-wong-cvmp.pdf).

This one-page report compares animation before and after programmatic pixelation
of human imagery. Its architecture follows reference, pose, and temporal conditioning.
The reported pixel-art treatment uses downsampling and nearest-neighbor enlargement.

**Implication:** its result does not demonstrate preservation of hand-authored pixel
art, exact palette constraints, or independent equipment layers. It is useful related
work, but insufficient as the basis for our implementation or training budget.

## Revised research decisions

These are project proposals drawn from the review, not results established by the papers.

1. **Keep the baseline's role narrow.** The current pixel-space U-Net checks whether
   data, alpha supervision, optimization, checkpoints, and serving inputs work. Its
   aligned references and one-body tests do not validate independently generated designs.
2. **Evaluate representations before diffusion training.** Round-trip real equipment
   through candidate RGBA encoders/decoders at native resolution and any proposed
   upscaled working resolution. Measure visible RGB, alpha, one-pixel boundaries,
   and composite error. Reject representations that destroy details before learning.
3. **Separate appearance from placement.** Compare a spatial/reference feature path
   against a lightweight image-prompt adapter; condition separately on the unmodified
   target body and available attachment/region information. Decide the pairing only
   after checking pretrained component compatibility and V100 memory use.
4. **Close the reference-domain gap explicitly.** Collect independent item drawings
   paired with fitted layers. A crop from the same animation is a useful easier
   benchmark, not a substitute. Test whether changing only the reference changes
   output design correctly; a model can memorize placements while ignoring the item.
5. **Add temporal modeling only after frame fitting works.** Compare the same test
   designs with and without temporal modules. Inspect detail loss, alignment, and
   visibility, including loop boundaries; do not optimize smoothness at the expense
   of preserving equipment or changing the supplied poses.
6. **Make evaluation equipment-specific.** Report per-design results, boundary quality,
   missing parts, wrong visibility, and color/detail fidelity. Use empty/copy-reference
   baselines. Composite similarity and whole-character identity scores can conceal
   errors in small equipment regions. Current code provides only some of these metrics.
7. **Freeze the supplied body operationally.** The model outputs equipment only, and
   the compositor uses original body pixels. A diffusion model's promise to preserve
   a background is not our mechanism for preserving the character.

## Still to investigate before choosing the research architecture

- Verify available pretrained weights, licenses, code paths, and cross-component
  compatibility for the chosen RGBA and reference-conditioning candidates.
- Inspect relevant architecture diagrams/qualitative failure figures directly and
  read the implementation details needed to reproduce the selected candidate.
- Design a paired independent-reference dataset and an attachment/occlusion test
  set; the current south-facing LPC walk does not cover behind-torso interactions.
- Establish the per-job GPU allocation and measure rather than infer training cost.

The research architecture remains undecided. The next HPC baseline run is a pipeline
diagnostic; these papers do not establish that the project-specific questions above
are already solved.
