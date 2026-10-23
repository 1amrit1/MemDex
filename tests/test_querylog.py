from memdex.querylog import count_log, log_query, read_log
from memdex.search import Hit


def _hit(rel="a.md", score=0.5):
    return Hit(score, rel, "x/" + rel, rel + " > h", 1, 2, "docs", "P", "t", "", False, "0:" + rel)


def test_log_and_read(tmp_cfg):
    _, load = tmp_cfg
    cfg = load()
    assert read_log(cfg) == [] and count_log(cfg) == 0
    log_query(cfg, "cli", "first question", [_hit()], 12.34, 5, collection="rules", project=None, superseded=False)
    log_query(cfg, "ui", "second about travel", [], 3.0, 10)
    log_query(cfg, "hook", "third question", [_hit("b.md", 0.7)], 8.0, 5)
    assert count_log(cfg) == 3

    newest = read_log(cfg, n=5)
    assert [e["query"] for e in newest] == ["third question", "second about travel", "first question"]
    assert newest[2]["filters"] == {"collection": "rules"} and newest[2]["ms"] == 12.3 and newest[2]["hits"][0]["rel_path"] == "a.md"
    assert [e["caller"] for e in read_log(cfg, n=5, caller="ui")] == ["ui"]
    assert [e["query"] for e in read_log(cfg, n=5, contains="QUESTION")] == ["third question", "first question"]
    assert len(read_log(cfg, n=1)) == 1
