import hashlib
import json

import pytest
import torch

from equipment_fitter.data import load_dataset
from equipment_fitter.fixture import create_fixture
from equipment_fitter.heldout import evaluate_protocol, validate_protocol
from equipment_fitter.model import EquipmentUNet


def frozen_fixture(tmp_path):
    full = create_fixture(tmp_path / 'data')
    doc = json.loads(full.read_text())
    dev = full.parent / 'dev.json'
    test = full.parent / 'test.json'
    dev.write_text(json.dumps({**doc, 'samples': [s for s in doc['samples'] if s['split'] == 'train']}))
    test.write_text(json.dumps({**doc, 'samples': [{**s, 'split': 'test'} for s in doc['samples'] if s['split'] == 'val']}))
    fingerprint = load_dataset(dev)[1]['fingerprint']
    checkpoint = tmp_path / 'last.pt'
    torch.save({'format_version': 1, 'input_contract': 'aligned-reference-rgba-v1',
                'config': {'width': 4, 'seed': 42}, 'step': 1, 'frame_size': doc['frame_size'],
                'dataset_fingerprint': fingerprint, 'model': EquipmentUNet(4).state_dict()}, checkpoint)
    return {'development_manifest': str(dev), 'test_manifest': str(test),
            'development_fingerprint': fingerprint, 'test_fingerprint': load_dataset(test)[1]['fingerprint'],
            'checkpoints': [{'name': 'unet-42', 'path': str(checkpoint), 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             'architecture': 'unet', 'seed': 42, 'step': 1}]}


def test_frozen_export_scores_exclude_reference_and_preserve_checkpoint(tmp_path):
    p = frozen_fixture(tmp_path)
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(p))
    result = evaluate_protocol(path, tmp_path / 'out')
    assert result['results'][0]['metrics']['model']['frames'] == 3
    assert result['results'][0]['metrics']['empty']['alpha_iou'] == 0
    validate_protocol(p)  # Checkpoint remains unchanged after inference.


def test_rejects_hash_and_split_tampering_before_inference(tmp_path):
    p = frozen_fixture(tmp_path)
    p['checkpoints'][0]['sha256'] = 'bad'
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_protocol(p)
    p['test_fingerprint'] = 'bad'
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        validate_protocol(p)


def test_rejects_design_family_overlap(tmp_path):
    from pathlib import Path
    p = frozen_fixture(tmp_path)
    path = Path(p['test_manifest'])
    test = json.loads(path.read_text())
    dev = json.loads(Path(p['development_manifest']).read_text())
    test['samples'][0]['design_id'] = dev['samples'][0]['design_id']
    path.write_text(json.dumps(test))
    p['test_fingerprint'] = load_dataset(path)[1]['fingerprint']
    with pytest.raises(ValueError, match='family overlap'):
        validate_protocol(p)
