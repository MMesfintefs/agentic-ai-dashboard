# FILE: tests/conftest.py
import os, time, pytest

# Keep Streamlit quiet during tests
os.environ.setdefault("STREAMLIT_ENV", "test")

@pytest.fixture(autouse=True)
def freeze_time(monkeypatch):
    # Deterministic timestamps for DB rows
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000)
    yield

@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    # Clear real keys so tests run offline
    for k in [
        "OPENAI_API_KEY","NEWS_API_KEY","EMAIL_HOST","EMAIL_USER","EMAIL_PASS",
        "EMAIL_PORT","EMAIL_SSL","EMAIL_FOLDER","GMAIL_SA_JSON","GMAIL_USER",
        "MS_GRAPH_TOKEN","GCAL_SA_JSON","GCAL_CAL_ID","GOOGLE_API_KEY",
        "MODEL_CHAT","MODEL_TTS","MODEL_STT","TIMEZONE","DB_PATH"
    ]:
        monkeypatch.delenv(k, raising=False)
    # Point the app DB at a temp file per test session
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    yield

@pytest.fixture
def app_module(monkeypatch):
    # Fresh import of app.py so env changes apply
    import importlib
    if "app" in list(importlib.sys.modules.keys()):
        del importlib.sys.modules["app"]
    import app
    return app

@pytest.fixture
def fake_openai(app_module, monkeypatch):
    # Force the app to use a fake OpenAI client
    from tests.stubs import FakeOpenAI
    monkeypatch.setattr(app_module, "_OPENAI_OK", True, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(app_module, "_openai_client", lambda: FakeOpenAI())
    return FakeOpenAI()

@pytest.fixture
def fake_ddg(monkeypatch):
    # Replace DuckDuckGo client with a deterministic fake
    from tests.stubs import FakeDDG
    import app
    monkeypatch.setattr(app, "DDGS", FakeDDG)
    return FakeDDG

@pytest.fixture
def fake_yfinance(monkeypatch):
    # Replace yfinance.Ticker with a fake that returns stable data
    from tests.stubs import FakeTicker
    import app
    monkeypatch.setattr(app.yf, "Ticker", FakeTicker)
    return FakeTicker
