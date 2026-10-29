# Equipment spritesheet fitting

Create an equipment design, then fit it across an existing character spritesheet.
The model outputs an aligned RGBA equipment sheet with occlusion baked into alpha.
The supplied character, poses, frame order, and timing remain unchanged.

Implementation language: **Python**, with **PyTorch** for training and inference.
Training will run on the University of Luxembourg's Iris GPU nodes; Aion can
support data preparation. The final product will expose design generation and
equipment fitting through a website.

See [the project plan](docs/PLAN.md) and [research notes](papers/README.md).
The [literature review](docs/LITERATURE_REVIEW.md) records reading coverage, limitations,
and the experiments required before selecting the research architecture.
Research PDFs remain in `papers/` locally; Git tracks their index and download
manifest rather than the large binary files. Datasets and model checkpoints also
remain outside Git.

## First runnable experiment

The data importer, validator, preview tool, and PyTorch baseline are implemented.
See [dataset preparation](docs/DATASET.md) and [training/inference](docs/TRAINING.md).
The initial reference is an aligned equipment frame from a known animation.
Independent generated equipment illustrations are a later research milestone.

The [Iris smoke-test instructions](hpc/README.md) explain data transfer, Python
automatic module/environment setup, job submission, and which results to return.
After transferring the prepared data, run `bash hpc/run.sh` inside your GPU
allocation, or submit from the project root:

```sh
mkdir -p logs
sbatch hpc/iris-smoke.sbatch
```

[CPU and Iris GPU results](docs/RESULTS.md) document what the baseline learned and where
held-out equipment still fails. Automated tests cover the pipeline and HPC launcher orchestration. The glove and held-out footwear pilots have also completed on an Iris V100.

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
