"""Development-only reference registration stress test; not independent artwork."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch

from .appearance import exported, metrics
from .data import asset_path, frame, load_dataset, rgba, save_json
from .inference import export_input
from .model import build_model
from .preview import preview_dataset
from .tensors import conditioning, image_tensor
from .training import environment, select_device

CONDITIONS = ('aligned', 'pair_shift', 'item_shift', 'centered_no_body')


def translate_visible(image, dx, dy):
    box = image.getchannel('A').getbbox()
    if box and (box[0] + dx < 0 or box[1] + dy < 0 or box[2] + dx > image.width or box[3] + dy > image.height):
        raise ValueError('Reference transform would clip visible pixels')
    result = Image.new('RGBA', image.size)
    result.paste(image, (dx, dy))
    return result


def transform_reference(reference, base, condition):
    if reference.size != base.size:
        raise ValueError('Reference pair dimensions differ')
    box = reference.getchannel('A').getbbox()
    if box is None:
        raise ValueError('Equipment reference is empty')
    if condition == 'aligned':
        return reference.copy(), base.copy()
    if condition == 'pair_shift':
        return translate_visible(reference, 8, 0), translate_visible(base, 8, 0)
    if condition == 'item_shift':
        return translate_visible(reference, 8, 0), base.copy()
    if condition == 'centered_no_body':
        dx = (reference.width - (box[2] - box[0])) // 2 - box[0]
        dy = (reference.height - (box[3] - box[1])) // 2 - box[1]
        return translate_visible(reference, dx, dy), Image.new('RGBA', base.size)
    raise ValueError('Unknown reference condition')


@torch.no_grad()
def run(protocol_path, output, requested_device='cpu'):
    protocol = json.loads(protocol_path.read_text())
    manifest = Path(protocol['development_manifest'])
    document, dataset = load_dataset(manifest)
    if dataset['fingerprint'] != protocol['development_fingerprint']:
        raise ValueError('Development fingerprint mismatch')
    if any(s['split'] == 'test' for s in document['samples']):
        raise ValueError('Reserved test samples are forbidden')
    samples = [s for s in document['samples'] if s['split'] == 'val']
    if any('reference' in s for s in samples):
        raise ValueError('Registration stress requires frame-derived references')
    if not samples:
        raise ValueError('No validation samples')
    if output.exists():
        raise ValueError('Choose a new output directory')
    for entry in protocol['checkpoints']:
        if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Frozen checkpoint hash mismatch')
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    device = select_device(requested_device)
    report = {'environment': environment(device), 'dataset': dataset,
              'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              'conditions': list(CONDITIONS), 'results': [],
              'scope': 'Registration stress on existing development designs. Transformed same-frame references are not independent illustrations. No training or test-set scoring.'}
    prepared = {}
    for sample in samples:
        source = output / 'references' / sample['id'] / 'source'
        export_input(manifest, sample['id'], source)
        ref, ref_base = rgba(source / 'reference.png'), rgba(source / 'reference-base.png')
        for condition in CONDITIONS:
            item, body = transform_reference(ref, ref_base, condition)
            folder = source.parent / condition
            folder.mkdir()
            item.save(folder / 'reference.png')
            body.save(folder / 'reference-base.png')
            save_json(folder / 'provenance.json', {'kind': 'derived_alignment_stress', 'condition': condition,
                      'source_reference_sha256': hashlib.sha256((source / 'reference.png').read_bytes()).hexdigest(),
                      'independently_drawn': False, 'target_frames_used_for_reference': False})
            prepared[sample['id'], condition] = (item, body)
    for entry in protocol['checkpoints']:
        checkpoint = torch.load(entry['path'], map_location='cpu', weights_only=True)
        if checkpoint['dataset_fingerprint'] != dataset['fingerprint']:
            raise ValueError('Checkpoint development dataset mismatch')
        model = build_model(checkpoint['config']).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        for condition in CONDITIONS:
            run_dir = output / entry['name'] / condition
            run_dir.mkdir(parents=True)
            pooled_predictions, pooled_targets, designs, preview_samples = [], [], {}, []
            for sample in samples:
                base = rgba(asset_path(manifest.parent, sample['base']))
                target_sheet = rgba(asset_path(manifest.parent, sample['equipment']))
                item, body = prepared[sample['id'], condition]
                ref, ref_base = image_tensor(item), image_tensor(body)
                prediction_sheet = Image.new('RGBA', base.size)
                predictions, targets = [], []
                for i, spec in enumerate(sample['frames']):
                    base_frame = image_tensor(frame(base, sample, i, document['frame_size']))
                    prediction = exported(model(conditioning(base_frame, ref_base, ref)[None].to(device)).sigmoid()[0]).cpu()
                    pixels = (prediction.permute(1, 2, 0).numpy() * 255).round().astype('uint8')
                    prediction_sheet.paste(Image.fromarray(pixels), tuple(spec['xy']))
                    if i != sample['reference_frame']:
                        predictions.append(prediction)
                        targets.append(exported(image_tensor(frame(target_sheet, sample, i, document['frame_size']))))
                prediction_sheet.save(run_dir / (sample['id'] + '.png'))
                base.save(run_dir / (sample['id'] + '-base.png'))
                designs[sample['id']] = metrics(predictions, targets)
                pooled_predictions.extend(predictions)
                pooled_targets.extend(targets)
                preview_samples.append({**sample, 'base': sample['id'] + '-base.png', 'equipment': sample['id'] + '.png'})
            save_json(run_dir / 'dataset.json', {**document, 'samples': preview_samples})
            preview_dataset(run_dir / 'dataset.json', run_dir / 'previews', allow_empty=True)
            report['results'].append({'name': entry['name'], 'architecture': entry['architecture'],
                                     'seed': entry['seed'], 'checkpoint_sha256': entry['sha256'], 'condition': condition,
                                     'metrics': metrics(pooled_predictions, pooled_targets), 'designs': designs})
            save_json(output / 'reference-stress.json', report)
    print(f'Reference stress complete: {output}/reference-stress.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    run(args.protocol, args.output, args.device)
