# FILE: tests/test_tools.py
import importlib
import sqlite3

def test_stocks_lookup(app_module, fake_yfinance):
    info = app_module.stocks_lookup("TSLA", "6mo")
    assert info["ticker"] == "TSLA"
    assert info["price"] == 103.0
    assert round(info["pct"], 2) == 3.00
    assert "history" in info and len(info["history"]) == 2

def test_news_fallback_ddg(app_module, fake_ddg, monkeypatch):
    # Ensure fallback path (no NEWS_API_KEY)
    monkeypatch.delenv("NEWS_API_KEY", raising=False)
    hs = app_module.news_headlines("tesla", 3)
    assert len(hs) == 3
    assert all("title" in h and "url" in h for h in hs)

def test_news_newsapi_happy(monkeypatch):
    # Force NewsAPI branch and stub network call
    class Resp:
        def __init__(self, js): self._js = js
        def raise_for_status(self): pass
        def json(self): return self._js
    def fake_get(url, params=None, timeout=12):
        return Resp({"articles":[
            {"title":"N1","url":"https://news/1"},
            {"title":"N2","url":"https://news/2"},
            {"title":"N3","url":"https://news/3"},
        ]})
    monkeypatch.setenv("NEWS_API_KEY", "x")
    import app as app1
    importlib.reload(app1)
    monkeypatch.setattr(app1.requests, "get", fake_get)
    hs = app1.news_headlines("anything", 2)
    assert [h["title"] for h in hs] == ["N1","N2"]

def test_places_google_and_fallback(monkeypatch, fake_ddg):
    # Google Places path
    class Resp:
        def __init__(self, js): self._js = js
        def raise_for_status(self): pass
        def json(self): return self._js
    def fake_get(url, params=None, timeout=10):
        return Resp({"results":[
            {"name":"Cafe A","place_id":"pid1","formatted_address":"addr 1"},
            {"nam
