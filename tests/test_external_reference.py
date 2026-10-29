import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
import torch

from equipment_fitter.data import load_dataset, frame, rgba
from equipment_fitter.fixture import create_fixture
from equipment_fitter.inference import export_input
from equipment_fitter.tensors import FrameDataset
from equipment_fitter.pilot import run_pilot


def prepare(tmp_path):
    path = create_fixture(tmp_path / 'data')
    doc = json.loads(path.read_text())
    # Synthetic test-only reference; not a real independently authored benchmark.
    image = Image.new('RGBA', (32, 32))
    image.paste((10, 90, 200, 255), (8, 10, 13, 17))
    image.save(path.parent / 'reference.png')
    provenance = {'creator': 'test fixture', 'source': 'synthetic test', 'rights': 'test only',
                  'design_description': 'Test color patch', 'method': 'human_drawn',
                  'image_sha256': hashlib.sha256((path.parent / 'reference.png').read_bytes()).hexdigest(),
                  'pairing_review': {'status': 'accepted', 'reviewer': 'test fixture', 'notes': 'Synthetic software test only'}}
    (path.parent / 'reference.json').write_text(json.dumps(provenance))
    doc['samples'][0]['reference'] = {'kind': 'separately_authored', 'image': 'reference.png', 'provenance': 'reference.json'}
    path.write_text(json.dumps(doc))
    return path, doc


def test_external_reference_used_fingerprinted_and_exported_without_body(tmp_path):
    path, doc = prepare(tmp_path)
    before = load_dataset(path)[1]['fingerprint']
    dataset = FrameDataset(path, 'train', 'canonical_transfer')
    assert dataset[0][2][2].max() == 200 / 255
    export_input(path, doc['samples'][0]['id'], tmp_path / 'input')
    assert not (tmp_path / 'input/reference-base.png').exists()
    assert (tmp_path / 'input/reference.png').read_bytes() == (path.parent / 'reference.png').read_bytes()
    metadata = json.loads((path.parent / 'reference.json').read_text())
    metadata['pairing_review']['notes'] = 'Updated review'
    (path.parent / 'reference.json').write_text(json.dumps(metadata))
    assert load_dataset(path)[1]['fingerprint'] != before
    with pytest.raises(ValueError, match='canonical_transfer'):
        FrameDataset(path, 'train')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'architecture': 'canonical_transfer', 'width': 4, 'steps': 1, 'threads': 1}))
    run_pilot(path, config, tmp_path / 'pilot', 'cpu')
    assert (tmp_path / 'pilot/previews/glove-0.png').exists()


def test_rejects_pending_pair_and_changed_image(tmp_path):
    path, _ = prepare(tmp_path)
    meta = path.parent / 'reference.json'
    info = json.loads(meta.read_text()); info['pairing_review']['status'] = 'pending'
    meta.write_text(json.dumps(info))
    with pytest.raises(ValueError, match='accepted pairing'):
        load_dataset(path)
    Image.new('RGBA', (32, 32), 'red').save(path.parent / 'reference.png')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_dataset(path)


def test_rejects_target_frame_masquerading_as_external_reference(tmp_path):
    path, doc = prepare(tmp_path)
    sample = doc['samples'][0]
    image = frame(rgba(path.parent / sample['equipment']), sample, 1, doc['frame_size'])
    image.save(path.parent / 'reference.png')
    meta = path.parent / 'reference.json'; info = json.loads(meta.read_text())
    info['image_sha256'] = hashlib.sha256((path.parent / 'reference.png').read_bytes()).hexdigest()
    meta.write_text(json.dumps(info))
    with pytest.raises(ValueError, match='duplicates a target'):
        load_dataset(path)


def test_authoring_draft_is_not_silently_accepted(tmp_path):
    from equipment_fitter.authoring import prepare as prepare_authoring
    path = create_fixture(tmp_path / 'targets')
    draft = prepare_authoring(path, tmp_path / 'dossier')
    assert (draft.parent / 'target-previews/glove-0.png').exists()
    metadata = json.loads((draft.parent / 'references/glove-0.json').read_text())
    assert metadata['pairing_review']['status'] == 'pending'
    with pytest.raises((ValueError, OSError)):
        load_dataset(draft)
