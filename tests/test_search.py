from memdex import store
from memdex.search import Hit, as_context, as_rules_text, as_table, build_where, search


def test_build_where():
    assert build_where() == {"superseded": False}
    assert build_where(include_superseded=True) is None
    assert build_where(collection="rules", include_superseded=True) == {"collection": "rules"}
    assert build_where(collection="rules", project="P1") == {"$and": [{"collection": "rules"}, {"project": "P1"}, {"superseded": False}]}


def _seed(tmp_path, emb):
    client = store.open_client(tmp_path / ".index")
    coll = store.reset_collection(client)
    docs = {
        "a": ("alpha beta gamma rollback capture", {"collection": "rules", "project": "P1", "superseded": False, "rel_path": "CLAUDE.md", "heading_path": "CLAUDE.md > 1.3", "start_line": 1, "end_line": 3, "path": "x/CLAUDE.md"}),
        "b": ("delta epsilon zeta budget travel", {"collection": "references", "project": "P2", "superseded": False, "rel_path": "trip.md", "heading_path": "trip.md > plan", "start_line": 5, "end_line": 9, "path": "x/trip.md"}),
        "c": ("alpha beta old superseded text", {"collection": "rules", "project": "P1", "superseded": True, "rel_path": "OLD.md", "heading_path": "OLD.md > x", "start_line": 1, "end_line": 1, "path": "x/OLD.md"}),
    }
    coll.add(ids=list(docs), documents=[d for d, _ in docs.values()], metadatas=[m for _, m in docs.values()], embeddings=emb.embed([d for d, _ in docs.values()]))
    return coll


def test_search_ranks_filters_and_hides_superseded(tmp_path, fake_embedder):
    coll = _seed(tmp_path, fake_embedder)
    hits = search(fake_embedder, coll, "alpha beta gamma rollback capture", k=5)
    assert hits[0].rel_path == "CLAUDE.md" and hits[0].score > 0.99
    assert all(not h.superseded for h in hits)  # "c" hidden by default

    with_old = search(fake_embedder, coll, "alpha beta", k=5, include_superseded=True)
    assert {h.rel_path for h in with_old} == {"CLAUDE.md", "trip.md", "OLD.md"}

    only_p2 = search(fake_embedder, coll, "alpha beta", k=5, project="P2")
    assert [h.rel_path for h in only_p2] == ["trip.md"]

    only_rules = search(fake_embedder, coll, "budget travel", k=5, collection="rules")
    assert [h.rel_path for h in only_rules] == ["CLAUDE.md"]
    assert search(fake_embedder, coll, "   ", k=5) == []


def test_formatters():
    h = Hit(0.4321, "CLAUDE.md", "x/CLAUDE.md", "CLAUDE.md > PART 1 > 1.3", 10, 20, "rules", "P1", "CLAUDE.md > PART 1 > 1.3\n\nbody", "2026-08-11", False, "0:CLAUDE.md#0")
    ctx = as_context([h], "q?")
    # heading_path already begins with rel_path, so the file name is printed once, not twice (2026-10-20)
    assert "## MemDex results for: q?" in ctx and "### 1. CLAUDE.md > PART 1 > 1.3" in ctx and "lines 10-20" in ctx and "project P1" in ctx
    assert "CLAUDE.md > CLAUDE.md" not in ctx
    assert "no results" in as_context([], "q?")
    # h.text opens with the heading path (as every ingested chunk does), so the header is the score alone
    rules_text = as_rules_text([h], "q?")
    assert "Rules from MemDex" in rules_text and "--- score 0.43 ---" in rules_text
    assert rules_text.count("CLAUDE.md > PART 1 > 1.3") == 1  # cited once, not twice
    # a hit whose body does NOT carry the path still gets a citation, printed in full
    bare = Hit(0.4321, "CLAUDE.md", "x/CLAUDE.md", "CLAUDE.md > PART 1 > 1.3", 10, 20, "rules", "P1", "body", "2026-08-11", False, "0:CLAUDE.md#0")
    assert "--- CLAUDE.md > PART 1 > 1.3 (score 0.43) ---" in as_rules_text([bare], "q?")
    assert as_table([h]).startswith("1. 0.432  rules      CLAUDE.md:10-20  | 1.3 (P1)")
