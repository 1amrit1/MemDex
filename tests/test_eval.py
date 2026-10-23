import pytest

from memdex.eval import EvalError, Expectation, Question, load_questions, matches, run_eval, save_result, load_last
from memdex.search import Hit


def _hit(rel, heading, score=0.5):
    return Hit(score, rel, "x/" + rel, heading, 1, 2, "rules", "P", "t", "", False, "0:" + rel)


def test_matches_path_and_heading():
    h = _hit("PROJECTS/MemDex_RAG.md", "MemDex_RAG.md > 3. Design > 3.7 Stack")
    assert matches(h, Expectation("PROJECTS/MemDex_RAG.md"))
    assert matches(h, Expectation("memdex_rag.md"))  # suffix + case-insensitive
    assert matches(h, Expectation("MemDex_RAG.md", heading="3.7"))
    assert not matches(h, Expectation("MemDex_RAG.md", heading="4.1"))
    assert not matches(h, Expectation("RAG.md"))  # must be a whole path component


def test_run_eval_recall_and_mrr():
    qs = [
        Question("a", (Expectation("a.md"),)),
        Question("b", (Expectation("b.md", "h2"),), collection="rules"),
        Question("c", (Expectation("missing.md"),)),
    ]
    answers = {
        "a": [_hit("a.md", "a.md > h1")],
        "b": [_hit("x.md", "x"), _hit("b.md", "b.md > h2")],
        "c": [_hit("y.md", "y")],
    }
    seen = []

    def fn(q, k, collection, project):
        seen.append((q, k, collection, project))
        return answers[q]

    r = run_eval(qs, fn, k=5)
    assert r.n == 3 and r.hits == 2 and r.recall == round(2 / 3, 4) and r.mrr == round((1 + 0.5) / 3, 4)
    assert r.rows[1]["rank"] == 2 and r.rows[2]["hit"] is False
    assert seen[1] == ("b", 5, "rules", None)


def test_load_questions_and_save(tmp_cfg, tmp_path):
    _, load = tmp_cfg
    cfg = load()
    f = tmp_path / "q.yaml"
    f.write_text("k: 3\nquestions:\n  - q: one\n    expect: [a.md]\n  - q: two\n    collection: rules\n    expect:\n      - {rel_path: b.md, heading: H}\n", encoding="utf-8")
    k, qs = load_questions(f)
    assert k == 3 and qs[1].collection == "rules" and qs[1].expect == (Expectation("b.md", "H"),)
    for bad in ("questions: []\n", "questions:\n  - q: x\n", "questions:\n  - expect: [a.md]\n"):
        f.write_text(bad, encoding="utf-8")
        with pytest.raises(EvalError):
            load_questions(f)
    with pytest.raises(EvalError):
        load_questions(tmp_path / "nope.yaml")

    r = run_eval(qs, lambda q, k, c, p: [], k)
    assert load_last(cfg) is None
    save_result(cfg, r)
    assert load_last(cfg)["n"] == 2 and load_last(cfg)["recall_at_k"] == 0.0
