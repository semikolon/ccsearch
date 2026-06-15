import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mannaminne as m


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class ChunkingTests(unittest.TestCase):
    def test_char_chunk_overlap_and_limit(self):
        chunks = list(m.chunk("abcdefghij", size=4, overlap=1, max_chunks=3))
        self.assertEqual(chunks, [(0, "abcd"), (1, "defg"), (2, "ghij")])

    def test_markdown_chunks_repeat_heading_path(self):
        text = "# Alpha\n" + ("one two three four five " * 20) + "\n## Beta\nBeta body"
        chunks = list(m.chunk_markdown(text, size=90, overlap=10, max_chunks=20))
        alpha_chunks = [c for _, c in chunks if "one two" in c or "three four" in c]
        self.assertTrue(alpha_chunks)
        self.assertTrue(all(c.startswith("Heading: Alpha\n\n") for c in alpha_chunks))
        self.assertTrue(any(c.startswith("Heading: Alpha > Beta\n\n") for _, c in chunks))

    def test_rows_strip_nuls_and_hash_chunk_text(self):
        rows = list(m._rows("doc", "doc:x", "proj", "T\x00itle", "a\x00b", "",
                            chunker=lambda _full: [(0, "a\x00b")]))
        self.assertEqual(rows[0][5], "Title")
        self.assertEqual(rows[0][6], "ab")
        self.assertEqual(rows[0][8], m.h("ab"))


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.old = (
            m._EMBED_URL_CACHE, m.EXPLICIT_EMBED_URL, m.Z4_EMBED_URL,
            m.DARWIN_EMBED_URL, m.EMBED_DIM, m.EMBED_PROBE_TIMEOUT,
        )
        m._EMBED_URL_CACHE = None
        m.EXPLICIT_EMBED_URL = None
        m.Z4_EMBED_URL = "http://z4.local/v1/embeddings"
        m.DARWIN_EMBED_URL = "http://darwin.local/v1/embeddings"
        m.EMBED_DIM = 3
        m.EMBED_PROBE_TIMEOUT = 0.5

    def tearDown(self):
        (
            m._EMBED_URL_CACHE, m.EXPLICIT_EMBED_URL, m.Z4_EMBED_URL,
            m.DARWIN_EMBED_URL, m.EMBED_DIM, m.EMBED_PROBE_TIMEOUT,
        ) = self.old

    def test_embed_batch_prefers_z4_when_available(self):
        calls = []

        def fake_urlopen(req, timeout):
            calls.append((req.full_url, timeout, json.loads(req.data.decode())))
            return FakeResponse({"data": [{"embedding": [1, 2, 3, 4]}]})

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            self.assertEqual(m._embed_batch(["hello"]), [[1, 2, 3]])

        self.assertEqual(calls[0][0], "http://z4.local/v1/embeddings")
        self.assertEqual(m._EMBED_URL_CACHE, "http://z4.local/v1/embeddings")

    def test_embed_batch_falls_back_to_darwin(self):
        calls = []

        def fake_urlopen(req, timeout):
            calls.append(req.full_url)
            if req.full_url == "http://z4.local/v1/embeddings":
                raise TimeoutError("z4 unavailable")
            return FakeResponse({"data": [{"embedding": [4, 5, 6, 7]}]})

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            self.assertEqual(m._embed_batch(["hello"]), [[4, 5, 6]])

        self.assertEqual(calls, ["http://z4.local/v1/embeddings", "http://darwin.local/v1/embeddings"])
        self.assertEqual(m._EMBED_URL_CACHE, "http://darwin.local/v1/embeddings")

    def test_embed_batch_retries_z4_after_darwin_cache(self):
        m._EMBED_URL_CACHE = "http://darwin.local/v1/embeddings"
        calls = []

        def fake_urlopen(req, timeout):
            calls.append(req.full_url)
            return FakeResponse({"data": [{"embedding": [7, 8, 9, 10]}]})

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            self.assertEqual(m._embed_batch(["hello"]), [[7, 8, 9]])

        self.assertEqual(calls, ["http://z4.local/v1/embeddings"])
        self.assertEqual(m._EMBED_URL_CACHE, "http://z4.local/v1/embeddings")

    def test_embed_pairs_splits_failed_batches(self):
        def fake_embed(texts):
            if len(texts) > 1:
                raise RuntimeError("batch too large")
            return [[len(texts[0])]]

        with mock.patch.object(m, "_embed_batch", fake_embed):
            out = m._embed_pairs([("a", "hello"), ("b", "world!")])

        self.assertEqual(out, [("a", [5]), ("b", [6])])


class RankingAndEvalTests(unittest.TestCase):
    def row(self, title):
        return ("id", "doc", "proj", title, "text", "")

    def source_row(self, source_id, title):
        return (f"{source_id}#0", "doc", "proj", title, "text", "")

    def test_query_terms_drop_filler_and_soft_terms_drop_generic_search_words(self):
        self.assertEqual(
            m._query_terms("what did I say about local codebase vector search?"),
            ["local", "codebase", "vector", "search"],
        )
        self.assertEqual(
            m._soft_terms("local codebase vector search"),
            ["codebase", "vector"],
        )

    def test_fusion_keeps_exact_keyword_above_semantic_only(self):
        ranked = m._fuse_ranked({
            "sem": {"r": self.row("semantic generic"), "kw": False, "sem": 0.99, "sem_rank": 1},
            "kw": {"r": self.row("exact needle"), "kw": True, "sem": 0.2, "kw_rank": 1, "exact": True},
        }, limit=2)
        self.assertEqual(ranked[0]["r"][3], "exact needle")

    def test_fusion_deduplicates_source_objects(self):
        ranked = m._fuse_ranked({
            "doc:one#0": {"r": self.source_row("doc:one", "first chunk"), "kw": True, "kw_rrf": 0.2},
            "doc:one#1": {"r": self.source_row("doc:one", "second chunk"), "kw": True, "kw_rrf": 0.1},
            "doc:two#0": {"r": self.source_row("doc:two", "other source"), "kw": True, "kw_rrf": 0.05},
        }, limit=3)
        self.assertEqual([r["r"][3] for r in ranked], ["first chunk", "other source"])

    def test_expectation_matching_supports_strings_and_fields(self):
        result = {"r": ("id:1", "doc", "dotfiles", "CLAUDE.md", "mannaminne canonical", "")}
        self.assertTrue(m._expectation_matches(result, "canonical"))
        self.assertTrue(m._expectation_matches(result, {"title": "CLAUDE.md", "project": "dotfiles"}))
        self.assertFalse(m._expectation_matches(result, {"title": "other"}))


if __name__ == "__main__":
    unittest.main()
