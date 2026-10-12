from pdfsum import cache, config


def test_cache_key_changes_with_every_setting():
    base = cache.cache_key("abc", "map_reduce", "mistral", 1000)
    assert base != cache.cache_key("abd", "map_reduce", "mistral", 1000)
    assert base != cache.cache_key("abc", "refine", "mistral", 1000)
    assert base != cache.cache_key("abc", "map_reduce", "llama3.1", 1000)
    assert base != cache.cache_key("abc", "map_reduce", "mistral", 2000)


def test_cache_key_is_filename_safe():
    key = cache.cache_key("abc", "map_reduce", "llama3.1:8b/instruct", 1000)
    assert not any(ch in key for ch in ':/\\ ')


def test_file_hash_is_stable_and_content_based():
    assert cache.file_hash(b"hello") == cache.file_hash(b"hello")
    assert cache.file_hash(b"hello") != cache.file_hash(b"hellp")


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path / "c"))
    payload = {"summary": "héllo", "seconds": 1.5}
    cache.save("k1", payload)
    assert cache.load("k1") == payload


def test_load_missing_or_corrupt_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    assert cache.load("nope") is None
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert cache.load("bad") is None
