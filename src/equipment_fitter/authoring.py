"""Prepare pending reference-pair dossiers; never label missing artwork as accepted."""
import argparse
from pathlib import Path
import shutil

from .data import load_dataset, save_json
from .preview import preview_dataset


def prepare(manifest, output):
    document, report = load_dataset(manifest)
    if output.exists():
        raise ValueError('Choose a new authoring directory')
    if output.resolve().is_relative_to(manifest.parent.resolve()):
        raise ValueError('Authoring output must be outside the source dataset')
    output.mkdir(parents=True)
    # Keep all source credits, pinned selections, and provenance alongside targets.
    shutil.copytree(manifest.parent, output / 'targets')
    samples = []
    for sample in document['samples']:
        name = sample['id']
        save_json(output / 'references' / f'{name}.json', {
            'creator': '', 'source': '', 'rights': '', 'design_description': '',
            'method': '', 'image_sha256': '',
            'pairing_review': {'status': 'pending', 'reviewer': '', 'notes': ''},
            'target_design_id': sample['design_id'], 'target_fingerprint': report['fingerprint']})
        samples.append({**sample, 'base': 'targets/' + sample['base'], 'equipment': 'targets/' + sample['equipment'],
                        'reference': {'kind': 'separately_authored', 'image': f'references/{name}.png',
                                      'provenance': f'references/{name}.json'}})
    save_json(output / 'draft-dataset.json', {**document, 'samples': samples})
    if (manifest.parent / 'CREDITS.csv').exists():
        shutil.copyfile(manifest.parent / 'CREDITS.csv', output / 'CREDITS.csv')
    preview_dataset(manifest, output / 'target-previews')
    (output / 'REVIEW.md').write_text(
        '# Pending reference pairs\n\n'
        'This draft is deliberately not trainable. Add separately drawn reference PNGs at the frame size; '
        'do not copy or transform target-frame pixels. Complete each references/*.json record with '
        'creator, source/prompt, rights, method, image SHA-256 and design description.\n\n'
        'Compare item identity, silhouette, palette, construction details, orientation and number of '
        'components against target-previews. A category match is insufficient. Record mismatches and '
        'reject or commission corrected references/targets. Mark accepted only after an actual pairing '
        'review; the software cannot prove authorship or semantic correspondence.\n\n'
        'Run equipment_fitter validate draft-dataset.json before any training. Keep recolors/related '
        'designs in one family/split. Existing evaluation sets are not fresh evidence for later changes.\n')
    return output / 'draft-dataset.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.manifest, args.output))
