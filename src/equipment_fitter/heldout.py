"""Evaluate hash-frozen checkpoints on reserved designs, without training."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .appearance import exported, metrics
from .data import asset_path, frame, load_dataset, rgba, save_json
from .inference import export_input, infer
from .preview import preview_dataset
from .tensors import image_tensor


def validate_protocol(protocol):
    dev_path, test_path = map(Path, (protocol['development_manifest'], protocol['test_manifest']))
    dev, dev_report = load_dataset(dev_path)
    test, test_report = load_dataset(test_path)
    if dev_report['fingerprint'] != protocol['development_fingerprint'] or test_report['fingerprint'] != protocol['test_fingerprint']:
        raise ValueError('Frozen dataset fingerprint mismatch')
    if any(s['split'] == 'test' for s in dev['samples']) or any(s['split'] != 'test' for s in test['samples']):
        raise ValueError('Development/test split violation')
    if {s['design_id'] for s in dev['samples']} & {s['design_id'] for s in test['samples']}:
        raise ValueError('Design family overlap with reserved test')
    names = set()
    if not protocol['checkpoints']:
        raise ValueError('No frozen checkpoints')
    for entry in protocol['checkpoints']:
        if entry['name'] in names or not entry['name'].replace('-', '').replace('_', '').isalnum():
            raise ValueError('Invalid or duplicate checkpoint name')
        names.add(entry['name'])
        path = Path(entry['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Frozen checkpoint hash mismatch')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['dataset_fingerprint'] != dev_report['fingerprint']:
            raise ValueError('Checkpoint training dataset mismatch')
        if (checkpoint['config'].get('architecture', 'unet') != entry['architecture'] or
                checkpoint['config']['seed'] != entry['seed'] or checkpoint['step'] != entry['step']):
            raise ValueError('Frozen checkpoint configuration mismatch')
        if checkpoint['frame_size'] != test['frame_size']:
            raise ValueError('Test frame size mismatch')
    return test, test_report


def evaluate_protocol(protocol_path, output, device='cpu'):
    protocol = json.loads(protocol_path.read_text())
    test, test_report = validate_protocol(protocol)  # Verify everything before inference.
    if output.exists():
        raise ValueError('Choose a new output directory')
    output.mkdir(parents=True)
    save_json(output / 'protocol.json', protocol)
    test_path = Path(protocol['test_manifest'])
    report = {'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              'dataset': test_report, 'results': [],
              'scope': 'Frozen checkpoints, reserved designs; aligned reference per view, same body and walk. No training or checkpoint selection.'}
    preview_dataset(test_path, output / 'ground-truth-previews')
    for entry in protocol['checkpoints']:
        run = output / entry['name']
        pooled = {name: [] for name in ('model', 'copy_reference', 'empty', 'target')}
        designs, samples = {}, []
        for sample in test['samples']:
            inputs, predictions = run / 'inputs' / sample['id'], run / 'predictions' / sample['id']
            export_input(test_path, sample['id'], inputs)
            infer(Path(entry['path']), inputs / 'base.png', inputs / 'reference.png',
                  inputs / 'reference-base.png', inputs / 'layout.json', predictions, device, 4, 'binary')
            predicted = rgba(predictions / 'equipment.png')
            target = rgba(asset_path(test_path.parent, sample['equipment']))
            reference = exported(image_tensor(rgba(inputs / 'reference.png')))
            values = {name: [] for name in pooled}
            for i in range(len(sample['frames'])):
                if i == sample['reference_frame']:
                    continue
                truth = exported(image_tensor(frame(target, sample, i, test['frame_size'])))
                values['target'].append(truth)
                values['model'].append(image_tensor(frame(predicted, sample, i, test['frame_size'])))
                values['copy_reference'].append(reference)
                values['empty'].append(torch.zeros_like(truth))
            designs[sample['id']] = {name: metrics(v, values['target']) for name, v in values.items() if name != 'target'}
            for name in pooled:
                pooled[name].extend(values[name])
            samples.append({**{k: v for k, v in sample.items() if k != 'reference'}, 'base': (inputs / 'base.png').relative_to(run).as_posix(),
                            'equipment': (predictions / 'equipment.png').relative_to(run).as_posix()})
        save_json(run / 'prediction-dataset.json', {**test, 'samples': samples})
        preview_dataset(run / 'prediction-dataset.json', run / 'previews', allow_empty=True)
        result = {**entry, 'designs': designs,
                  'metrics': {name: metrics(v, pooled['target']) for name, v in pooled.items() if name != 'target'}}
        save_json(run / 'scores.json', result)
        report['results'].append(result)
        save_json(output / 'heldout.json', report)
    print(f"Frozen test complete: {output}/heldout.json", flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    evaluate_protocol(args.protocol, args.output, args.device)
