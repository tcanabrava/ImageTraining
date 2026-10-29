import io
import json

import pytest
from PIL import Image

from equipment_fitter.data import load_dataset
from equipment_fitter.lpc import import_lpc


def test_import_retains_alpha_coordinates_and_credits(tmp_path, monkeypatch):
    config = {"revision": "a" * 40, "description": "test", "category": "glove", "animation": "walk",
              "view": "south", "frame_size": [8, 8], "duration_ms": 100, "reference_frame": 0,
              "source_frames": [[8, 0], [0, 0]], "base_layers": ["spritesheets/base.png"],
              "equipment": [{"id": "glove", "design_id": "glove", "split": "train", "path": "spritesheets/glove.png"}]}
    image = Image.new("RGBA", (16, 8))
    image.putpixel((1, 1), (255, 0, 0, 255))
    image.putpixel((9, 1), (0, 255, 0, 255))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    credits = b"filename,notes,authors,licenses,urls\nbase.png,,Artist,CC0,https://example.org\nglove.png,,Artist,CC0,https://example.org\n"
    monkeypatch.setattr("equipment_fitter.lpc.fetch_asset", lambda rev, name, cache: credits if name == "CREDITS.csv" else buf.getvalue())
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(config))
    manifest = import_lpc(selection, tmp_path / "out", tmp_path / "cache")
    assert load_dataset(manifest)[1]["target_frames"]["train"] == 1
    with Image.open(manifest.parent / "glove.png") as result:
        assert result.getpixel((1, 1)) == (0, 255, 0, 255)
        assert result.getpixel((9, 1)) == (255, 0, 0, 255)
    assert "Artist" in (manifest.parent / "CREDITS.csv").read_text()
    assert len(json.loads((manifest.parent / "provenance.json").read_text())["assets"]) == 2
    with pytest.raises(ValueError, match="already exists"):
        import_lpc(selection, tmp_path / "out", tmp_path / "cache")


def test_multiview_split_isolation_and_reserved_manifest(tmp_path, monkeypatch):
    from equipment_fitter.data import save_json
    from equipment_fitter import lpc
    original_import = lpc.import_lpc

    def fake_import(selection, output, cache):
        config = json.loads(selection.read_text())
        output.mkdir()
        Image.new('RGBA', (16, 8), (1, 2, 3, 255)).save(output / 'base.png')
        samples = []
        for item in config['equipment']:
            samples.append({'id': item['id'], 'design_id': item['design_id'], 'split': item['split'],
                            'category': 'footwear', 'base': 'base.png', 'equipment': 'base.png',
                            'reference_frame': 0, 'frames': [{'xy': [0, 0], 'duration_ms': 100},
                                                          {'xy': [8, 0], 'duration_ms': 100}]})
        save_json(output / 'dataset.json', {'schema_version': 1, 'frame_size': [8, 8], 'samples': samples})
        save_json(output / 'provenance.json', {})
        (output / 'CREDITS.csv').write_text('credits')
        return output / 'dataset.json'

    monkeypatch.setattr(lpc, 'import_lpc', fake_import)
    config = {'revision': 'a' * 40, 'frame_size': [8, 8], 'animation': 'walk', 'description': 'test',
              'views': [{'name': name, 'source_frames': [[0, 0], [8, 0]]} for name in ['north', 'south']],
              'equipment': [{'id': 'train', 'design_id': 'family-a', 'split': 'train'},
                            {'id': 'test', 'design_id': 'family-b', 'split': 'test'}]}
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps(config))
    full = original_import(selection, tmp_path / 'out', tmp_path / 'cache')
    assert load_dataset(full)[1]['target_frames'] == {'train': 2, 'val': 0, 'test': 2}
    dev, _ = load_dataset(full.parent / 'development.json')
    reserved, _ = load_dataset(full.parent / 'reserved-test.json')
    assert {s['split'] for s in dev['samples']} == {'train'}
    assert {s['split'] for s in reserved['samples']} == {'test'}
    assert len({s['id'] for s in dev['samples']}) == 2
    assert len({s['design_id'] for s in dev['samples']}) == 1
    config['equipment'][1]['design_id'] = 'family-a'
    selection.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='leakage'):
        original_import(selection, tmp_path / 'bad', tmp_path / 'cache')
    assert not (tmp_path / 'bad').exists()


def test_comparison_refuses_reserved_test_data(tmp_path):
    from equipment_fitter.fixture import create_fixture
    from equipment_fitter.compare import compare
    manifest = create_fixture(tmp_path / 'fixture')
    document = json.loads(manifest.read_text())
    for sample in document['samples']:
        if sample['split'] == 'val':
            sample['split'] = 'test'
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='reserved test'):
        compare(manifest, tmp_path / 'run', 'cpu')
    assert not (tmp_path / 'run').exists()
