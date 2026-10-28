# Research for layered, animated game sprites

Collected 2026-10-28. PDFs are from arXiv or an author's website. These papers cover components of the proposed system; they do not establish that arbitrary equipment can already be transferred perfectly across game sprite animations.

## Confirmed project scope

The product has two operations:

1. **Create equipment:** a prompt such as "draw a glove" produces an equipment reference for the user to approve.
2. **Fit equipment to an existing spritesheet:** the approved reference and a supplied character spritesheet produce a matching RGBA equipment spritesheet. Equipment follows the existing body poses, with appropriate placement, perspective, deformation, and visibility in every frame.

The supplied spritesheet defines all character movements. Character pixels, poses, frame layout, frame order, anchors, and timing remain unchanged. Creating base animations or new character movements is outside the project scope, including future expansion described here. Hidden equipment pixels are transparent, so playback draws the unchanged base frame followed by its equipment overlay.

The long-term target covers any equipment category, including gloves, boots, helmets, armor, weapons, and capes. This is a research objective, not an established generalization guarantee. Training must cover the relevant attachment and deformation behavior: gloves follow hands, rigid weapons retain their shape, and flexible equipment responds to the supplied animation. Begin with one category to validate the method, then expand equipment and base-spritesheet diversity.

The two operations do not require the same model. Equipment design generation can use a suitable existing generator; the central research task is preserving the approved design while fitting it across supplied frames. Pose/depth estimates or attachment annotations may help the fitting stage, but they describe existing poses rather than generate new ones.

## Initial asset sourcing

The user does not have a base spritesheet and intends to use freely available online spritesheets. No final restriction to pixel art or illustrated artwork has been chosen. Art style affects preprocessing, edge quality, resolution, and evaluation; one coherent style is a proposed first benchmark, not a permanent product restriction.

[Universal LPC](https://github.com/LiberatedPixelCup/Universal-LPC-Spritesheet-Character-Generator) is a candidate source because it provides compatible body and equipment assets. Its [generator](https://liberatedpixelcup.github.io/Universal-LPC-Spritesheet-Character-Generator/index.html) supports exports split by animation and item. Inspect a selected subset for coverage, alignment, and baked occlusion before treating it as ground-truth training data. Preserve per-asset provenance and licenses from `CREDITS.csv`; the collection has mixed licenses and is not uniformly CC0.

These assets could provide paired examples directly, reducing the initial dependence on synthetic 3D rendering. Extract a reference from one equipment frame and train fitting to other frames; distinguish this reference format from independently generated equipment illustrations and evaluate that gap separately. Split by equipment design before making crops, recolors, or frame examples, and later evaluate on different body templates and asset collections. Thousands of recolors and body/item combinations do not imply thousands of independent designs.

[Kenney](https://kenney.nl/support) is another source to inspect: its asset-page game assets are declared CC0, but suitability for matched equipment-animation pairs must be checked per pack. Free download availability alone does not describe an asset's license or make a flattened spritesheet a separated equipment training target.

## Suggested reading order

| Local PDF | Paper and source | Relevance |
| --- | --- | --- |
| [2024-layerdiffuse.pdf](2024-layerdiffuse.pdf) | [Transparent Image Layer Diffusion using Latent Transparency](https://arxiv.org/abs/2402.17113) | Native transparency and conditional layer generation. A starting point for actual RGBA output. |
| [2025-qwen-image-layered.pdf](2025-qwen-image-layered.pdf) | [Qwen-Image-Layered: Towards Inherent Editability via Layer Decomposition](https://arxiv.org/abs/2512.15603) | Decomposes an image into multiple RGBA layers. Useful for investigating separation of existing artwork, but decomposition alone does not provide animation or reliable equipment slots. |
| [2024-sprite-sheet-diffusion.pdf](2024-sprite-sheet-diffusion.pdf) | [Sprite Sheet Diffusion: Generate Game Character for Animation](https://arxiv.org/abs/2412.03685) | Directly relevant adaptation of reference, pose, and motion conditioning to game sprite animation. [Author repository](https://github.com/chenganhsieh/Sprite-Sheet-Diffusion). |
| [2025-sprite-sheet-generation.pdf](2025-sprite-sheet-generation.pdf) | [Sprite Sheet Generation using a Diffusion Model](https://marcovolino.github.io/docs/papers/2025-wong-cvmp.pdf), Hei Lit Wong and Marco Volino | One-page research poster comparing animation before versus after pixelation. Useful related experiment, not a full-length paper or proof of equipment-layer generation. |
| [2023-animate-anyone.pdf](2023-animate-anyone.pdf) | [Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation](https://arxiv.org/abs/2311.17117) | Reference appearance, target poses, and temporal modeling. Relevant architecture ideas; does not directly output interchangeable RGBA equipment sheets. |
| [2023-controlnet.pdf](2023-controlnet.pdf) | [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) | Conditions generation on spatial information such as pose, depth, and segmentation. |
| [2023-ip-adapter.pdf](2023-ip-adapter.pdf) | [IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://arxiv.org/abs/2308.06721) | Reference-image conditioning for appearance. Useful for studying identity and item-design retention, without guaranteeing exact reproduction. |
| [2023-text2layer.pdf](2023-text2layer.pdf) | [Text2Layer: Layered Image Generation using Latent Diffusion Model](https://arxiv.org/abs/2307.09781) | Earlier work jointly modeling foreground, background, mask, and composite. |
| [2024-layerdiff.pdf](2024-layerdiff.pdf) | [LayerDiff: Exploring Text-guided Multi-layered Composable Image Synthesis via Layer-Collaborative Diffusion Model](https://arxiv.org/abs/2403.11929) | Collaboration between generated layers. LayerDiff and LayerDiffuse are different projects. |
| [2026-lade.pdf](2026-lade.pdf) | [LaDe: Unified Multi-Layered Graphic Media Generation and Decomposition](https://arxiv.org/abs/2603.17965) | Joint layered graphic generation and decomposition; additional context for architectures that model multiple layers. |

Dates in filenames refer to initial publication/preprint years. Downloads use the current PDF at each source; `manifest.json` records the resolved URL, byte count, and SHA-256 for the actual saved file.

## Proposed experiment (engineering recommendation, not a published result)

Start with equipment transfer onto an existing, approved base animation. Inputs would be a base sprite sequence, an equipment reference, an equipment slot, and pose/depth information when available. The agreed output is one aligned RGBA equipment spritesheet for that specific base animation, with occlusion baked into alpha and identical canvas size, frame order, and anchor points to the base sequence.

1. Restrict the first experiment to one body template, one view, one animation, and one equipment category. Test unseen equipment designs rather than only new frames of training designs.
2. Build paired examples with a rigged 3D character and swappable equipment, rendered using fixed cameras and lighting. Export base frames, equipment colors, alpha/visibility masks, depth, poses, and the equipped composite. Supplement with artist-authored layered sprites to address the gap between rendered images and the desired art style.
3. Establish a deterministic rig-and-render or 2D-warp baseline before training. It may already solve the production need when equipment can be represented by geometry or attached artwork.
4. Adapt a pretrained image model with reference-image and spatial conditioning. Use a representation that explicitly supports alpha; ordinary RGB generation plus background removal is an alternative baseline, not equivalent ground-truth transparency. Train on short sequences with temporal interaction so frames share information.
5. Supervise equipment appearance, alpha, and recomposited images. Use correspondence-aware temporal losses only in regions with valid visibility; raw frame differences penalize legitimate movement. Ignore undefined RGB values in fully transparent pixels.
6. Evaluate equipment identity, alignment, occlusion, edge halos, temporal flicker, loop transitions, and untouched body pixels. Split validation by character/item identity and animation to prevent neighboring-frame leakage.

Occlusion is baked into the equipment layer: where the torso or hand hides a sword, the sword frame has zero alpha. Runtime rendering can therefore draw the base followed by a single equipment overlay. Separate front/back layers are not required for this agreed scope. The model still needs to infer visibility, and training targets must encode it correctly. Changing the body or pose may require regenerating the equipment overlay; this dependence is accepted. Shadows cast onto the body may require an additional effect layer. Removing equipment from an already dressed reference also requires reconstructing hidden body pixels, which segmentation alone cannot recover.

A single front-view reference does not reveal the back of an item. Supply additional views or accept inferred details. Pixel-art output adds palette, silhouette, and pixel-grid constraints that should be evaluated at the final game resolution.

Only after equipment transfer works should the experiment expand to additional equipment categories, body shapes, supplied animations, multiple views, and multiple interacting equipment slots. All character animation remains supplied input. Training data diversity and held-out performance should determine dataset expansion; no reliable training budget can be inferred from the concept alone.

## Available compute and revised execution plan

The user has access to the University of Luxembourg's Aion and Iris clusters. Their account's GPU allocation, concurrent-job limits, storage quota, and available GPU-hours are not yet established; facility totals must not be treated as a personal allocation.

Hardware documentation checked on 2026-10-28:

- [Aion compute nodes](https://hpc-docs.uni.lu/systems/aion/compute/): CPU nodes with 128 cores and 256 GB RAM each. Proposed role: parallel synthetic-data rendering on CPUs, preprocessing, dataset validation, and packaging.
- [Iris compute nodes](https://hpc-docs.uni.lu/systems/iris/compute/): GPU nodes with four NVIDIA V100 GPUs each, with either 16 GB or 32 GB per GPU. Proposed role: model training and GPU inference; prefer 32 GB devices for initial sequence experiments when available.

This access supports a more substantial research program: generate diverse paired data, compare architecture variants, and run repeated evaluations. Start by profiling a compact pretrained model on one GPU, then benchmark a single four-GPU node before considering multiple nodes. Standard data parallelism replicates the model; four 32 GB GPUs do not automatically provide a single 128 GB memory pool. Select model size, sequence length, precision, and attention implementations for the actual V100 environment.

Measure peak memory, training throughput, validation quality, and checkpoint size during a pilot. Use those measurements and the allocated GPU-hours to choose experiment sizes. Keep the initial problem narrow to make failures interpretable; expand data diversity and training scope after the single-slot baseline works. Access to a cluster makes custom conditioning and temporal-module training more practical, but does not by itself justify training a general image foundation model from scratch.

## Website deployment requirement

The user intends to make the trained models available through a website where people can run animations. This is a deployment requirement for the research, not a request to build or publish the website yet.

Confirmed interaction: a visitor asks for an equipment design, approves it, then asks the system to fit it to a supplied character spritesheet. The visitor previews the resulting layered animation and downloads the equipment sheet. The website never requests generation of base character animations or new character movements.

Separate model inference from animation playback:

1. The browser submits the equipment prompt to a design-generation service and presents the resulting reference for approval.
2. After approval, the browser submits the reference and supplied base spritesheet to an API, which creates an asynchronous equipment-fitting job.
3. A GPU worker loads a versioned trained model and generates an aligned RGBA equipment sequence with occlusion baked into alpha.
4. The service stores the spritesheets and an animation manifest, and returns their locations when the job completes.
5. The browser plays the animation by drawing matching base and equipment frames using Canvas or WebGL. Playback speed, pausing, and switching between already-generated compatible assets do not require further model inference.

The animation manifest should define frame rectangles, durations, anchors, animation names, loop behavior, and the base-animation version to which each overlay belongs. Preserve the alpha channel in downloadable assets. Cache generated outputs using inputs, preprocessing settings, generation settings, seed, and model version so repeated playback does not repeat generation.

Treat Aion/Iris as research compute. Production serving needs an explicit hosting arrangement; access to the clusters does not establish that they can host a public inference service. The default proposal is a separate GPU inference service behind a job queue. Browser-only inference is a separate deployment option that would require model-size, memory, compatibility, and latency measurements.

Evaluate deployment alongside visual quality from the first prototype: peak inference GPU memory, cold model load time, warm generation latency per animation, throughput, output size, and estimated cost per job on the chosen serving hardware. Aim initially for a model that can run on one serving GPU, with its memory budget selected before committing to an architecture. Export model weights, configuration, preprocessing, and the inference entry point independently of cluster-specific training scripts. Check pretrained-model and dataset terms for the intended public deployment before selecting them.

Equipment fitting runs once per requested design/base-animation combination, and the resulting sheets can be played repeatedly without model inference. Generation latency affects the wait for equipment fitting; it does not determine animation playback frame rate.
