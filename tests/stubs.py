# FILE: tests/stubs.py
import json
import pandas as pd

def _json(obj) -> str:
    return json.dumps(obj)

# ---------------------------
# Fake OpenAI client
# ---------------------------
class _Choice:
    def __init__(self, content: str):
        self.message = type("M", (), {"content": content})

class _Resp:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]

class _Completions:
    def create(self, model, messages, temperature=0.3, tools=None):
        # Minimal intent: detect planner vs summarizer vs generic
        sys = messages[0]["content"].lower()
        if "planner" in sys:
            plan = {
                "steps": [
                    {"tool": "stocks.lookup", "args": {"ticker": "TSLA", "period": "6mo"}, "note": "get price"},
                    {"tool": "news.search", "args": {"topic": "Tesla"}, "note": "headlines"},
                ]
            }
            return _Resp(_json(plan))
        if "summarize tool results" in sys or "summarize the tool results" in sys:
            return _Resp("Tesla up 3%. See 3 headlines.")
        # Generic chat fallback
        return _Resp("Stub response.")

class FakeOpenAI:
    """Drop-in stub for openai.OpenAI used by app._openai_client()."""
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _Completions()})
        self.audio = type(
            "Audio",
            (),
            {
                "speech": type("Speech", (), {"create": staticmethod(lambda **k: b"\x00\x01fake-mp3\x02")}),
                "transcriptions": type("Trans", (), {"create": staticmethod(lambda **k: type("R", (), {"text": "transcribed text"})())}),
            },
        )

# ---------------------------
# Fake DuckDuckGo client
# ---------------------------
class FakeDDG:
    """Context-manager compatible DDGS replacement with stable outputs."""
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False

    def news(self, query, max_results=8):
        return [
            {"title": "A", "url": "https://x/a"},
            {"title": "B", "url": "https://x/b"},
            {"title": "C", "url": "https://x/c"},
        ]

    def text(self, query, max_results=6):
        return [
            {"title": "P1", "href": "https://p/1", "body": "desc1"},
            {"title": "P2", "href": "https://p/2", "body": "desc2"},
        ]

# ---------------------------
# Fake yfinance.Ticker
# ---------------------------
class FakeTicker:
    """Returns a tiny, deterministic history: 100 -> 103 (+3%)."""
    def __init__(self, ticker): self.ticker = ticker
    def history(self, period="3mo", interval="1d"):
        return (
            pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                    "Close": [100.0, 103.0],
                }
            )
            .set_index("Date")
        )
