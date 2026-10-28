# Equipment spritesheet fitting

Create an equipment design, then fit it across an existing character spritesheet.
The model outputs an aligned RGBA equipment sheet with occlusion baked into alpha.
The supplied character, poses, frame order, and timing remain unchanged.

Implementation language: **Python**, with **PyTorch** for training and inference.
Training will run on the University of Luxembourg's Iris GPU nodes; Aion can
support data preparation. The final product will expose design generation and
equipment fitting through a website.

See [the project plan](docs/PLAN.md) and [research notes](papers/README.md).
Research PDFs remain in `papers/` locally; Git tracks their index and download
manifest rather than the large binary files. Datasets and model checkpoints also
remain outside Git.

## Scope

- Input: approved equipment reference plus an existing base animation.
- Output: matching equipment-only RGBA frames, suitable for drawing over the base.
- Long-term equipment coverage: gloves, armor, helmets, weapons, and flexible items.
- Character animation generation is outside scope.
- The initial benchmark uses existing layered sprite assets; arbitrary styles and
  independently generated item illustrations require separate validation.

## Development policy

Complete and verify each implementation milestone, then commit it. Commit messages
must not include a `Co-authored-by` trailer. Record measured results and limitations;
do not label a pipeline smoke test as demonstrated equipment generalization.
