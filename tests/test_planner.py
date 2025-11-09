# FILE: tests/test_planner.py
import sqlite3
import json

def test_planner_exec_happy_path(app_module, fake_openai, fake_yfinance, fake_ddg, monkeypatch):
    # Force NewsAPI off so DDG path triggers
    monkeypatch.delenv("NEWS_API_KEY", raising=False)

    out = app_module.planner_execute("Find Tesla’s performance and headlines.")
    assert isinstance(out, str) and out.strip() != ""

    # Task row persisted
    con = sqlite3.connect(app_module.SET.DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT COUNT(1) FROM tasks")
    cnt = cur.fetchone()[0]
    con.close()
    assert cnt == 1

def test_memory_written(app_module, fake_openai, fake_yfinance, fake_ddg):
    _ = app_module.planner_execute("Find Tesla’s performance and headlines.")
    mems = app_module.MEM.list(limit=10)
    assert any(m["kind"] == "short" for m in mems)

def test_plan_capped_steps(app_module, fake_openai, monkeypatch):
    # Swap in an OpenAI stub that emits 10 steps; we should execute at most 5
    class NoisyOpenAI:
        class _C:
            def create(self, model, messages, temperature=0.3, tools=None):
                steps = [{"tool":"news.search","args":{"topic":"x"}, "note":"n"} for _ in range(10)]
                return type("R", (), {
                    "choices":[type("C", (), {"message": type("M", (), {"content": json.dumps({"steps":steps})})})]
                })
        def __init__(self):
            self.chat = type("Chat", (), {"completions": self._C()})

    monkeypatch.setattr(app_module, "_openai_client", lambda: NoisyOpenAI())

    # Monkeypatch tool to count invocations
    calls = {"n": 0}
    def fake_news_headlines(topic, n=3):
        calls["n"] += 1
        return [{"title":"t","url":"u"}]

    monkeypatch.setattr(app_module, "news_headlines", fake_news_headlines)

    out = app_module.planner_execute("Do everything.")
    assert out.strip() != ""
    assert calls["n"] == 5  # capped at 5 steps
