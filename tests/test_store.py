import json

from fapeft.store import atomic_write_text, cached_json, read_json


def test_atomic_write_leaves_no_temp_file(tmp_path):
    p = tmp_path / "a" / "x.json"
    atomic_write_text(p, "{}")
    assert p.read_text() == "{}"
    assert [f.name for f in p.parent.iterdir()] == ["x.json"]


def test_read_json_missing_or_corrupt_is_none(tmp_path):
    assert read_json(tmp_path / "nope.json") is None
    (tmp_path / "bad.json").write_text('{"half": ')
    assert read_json(tmp_path / "bad.json") is None


def test_cached_json_computes_once(tmp_path):
    calls = []
    fn = lambda: calls.append(1) or {"v": 1}
    p = tmp_path / "r.json"
    assert cached_json(p, fn, log=lambda m: None) == {"v": 1}
    assert cached_json(p, fn, log=lambda m: None) == {"v": 1}
    assert len(calls) == 1
    assert json.loads(p.read_text()) == {"v": 1}


def test_corrupt_cache_is_recomputed(tmp_path):
    p = tmp_path / "r.json"
    p.write_text("{trunc")
    assert cached_json(p, lambda: {"v": 2}, log=lambda m: None) == {"v": 2}
