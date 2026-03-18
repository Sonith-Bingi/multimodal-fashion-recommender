from recommender.config import Settings


def test_default_settings() -> None:
    s = Settings()
    assert s.dense_k >= 2
    assert s.seq_len >= 1
    assert s.min_seq >= 2


def test_paths_are_constructed() -> None:
    s = Settings()
    assert s.notebook_path.name == "recotwotower.ipynb"
    assert s.index_path.name.endswith(".faiss")
    assert s.vectors_path.name.endswith(".npy")
