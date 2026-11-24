import numpy as np
import pytest

from saathi.retriever import HybridRetriever, rrf, tokenize


class StubEmbedder:
    """Deterministic fake embedder so tests never touch the network."""

    name = "stub"

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        v = np.array([[len(t) % 7 + 1.0, t.count("a") + 1.0, t.count("e") + 1.0] for t in texts])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_tokenize_drops_stopwords():
    assert tokenize("What is the PM-KISAN scheme?") == ["pm", "kisan", "scheme"]


def test_rrf_prefers_items_ranked_high_by_both():
    scores = rrf([[0, 1, 2], [1, 0, 2]])
    assert scores[0] == pytest.approx(scores[1]) and scores[0] > scores[2]


@pytest.mark.parametrize(
    "q,expected",
    [
        ("free LPG gas connection for poor women", "pmuy"),
        ("unemployment allowance for graduates", "ka-yuva-nidhi"),
        ("life insurance 436 premium", "pmjjby"),
        ("loan for street vendors", "pm-svanidhi"),
        ("savings account for my daughter", "ssy"),
    ],
)
def test_bm25_finds_right_scheme(offline_pipe, q, expected):
    assert expected in [c.scheme_id for c in offline_pipe.retrieve(q)][:3]


def test_hybrid_mode_and_index_cache(offline_pipe, tmp_path):
    emb = StubEmbedder()
    r1 = HybridRetriever(offline_pipe.chunks, emb, cache_dir=tmp_path)
    assert r1.mode == "hybrid" and len(r1.search("pension for old age", k=4)) == 4
    assert len(list(tmp_path.glob("index-*.npy"))) == 1

    emb2 = StubEmbedder()
    HybridRetriever(offline_pipe.chunks, emb2, cache_dir=tmp_path)
    assert emb2.calls == 0  # corpus vectors came from the on-disk cache


def test_broken_embedder_degrades_to_bm25(offline_pipe):
    class Broken:
        name = "broken"

        def embed(self, texts):
            raise OSError("download blocked")

    r = HybridRetriever(offline_pipe.chunks, Broken())
    assert r.mode == "bm25" and r.search("pension")
