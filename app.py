# FILE: app.py
# Run: streamlit run app.py

import os, io, re, time, json, base64, sqlite3, logging, datetime as dt
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import yfinance as yf
import requests
import streamlit as st
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings
from duckduckgo_search import DDGS

# Optional imports guarded by feature flags
try:
    from openai import OpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False

try:
    from google.oauth2.service_account import Credentials as SA_Credentials
    from googleapiclient.discovery import build as gbuild
    _GAPI_OK = True
except Exception:
    _GAPI_OK = False

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    _WEBRTC_OK = True
except Exception:
    _WEBRTC_OK = False


# ---------------- Settings & flags ----------------
class AppSettings(BaseSettings):
    OPENAI_API_KEY: str = Field(default="")
    NEWS_API_KEY: str = Field(default="")
    EMAIL_HOST: str = Field(default="")
    EMAIL_USER: str = Field(default="")
    EMAIL_PASS: str = Field(default="")
    EMAIL_PORT: int = Field(default=993)
    EMAIL_SSL: bool = Field(default=True)
    EMAIL_FOLDER: str = Field(default="INBOX")
    GMAIL_SA_JSON: str = Field(default="")
    GMAIL_USER: str = Field(default="")
    MS_GRAPH_TOKEN: str = Field(default="")
    GCAL_SA_JSON: str = Field(default="")
    GCAL_CAL_ID: str = Field(default="primary")
    GOOGLE_API_KEY: str = Field(default="")
    DB_PATH: str = Field(default="agentic_ai.db")
    MODEL_CHAT: str = Field(default="gpt-5-chat-latest")
    MODEL_TTS: str = Field(default="gpt-5-tts")
    MODEL_STT: str = Field(default="gpt-5-transcribe")

    class Config:
        case_sensitive = False


def load_settings() -> AppSettings:
    data = {}
    for k in AppSettings.__fields__:
        if k in st.secrets:
            data[k] = st.secrets[k]
        elif os.getenv(k) is not None:
            data[k] = os.getenv(k)
    try:
        return AppSettings(**data)
    except ValidationError as e:
        st.error(f"Settings validation error: {e}")
        return AppSettings()


SET = load_settings()
FEAT = {
    "openai": _OPENAI_OK and bool(SET.OPENAI_API_KEY),
    "newsapi": bool(SET.NEWS_API_KEY),
    "imap": bool(SET.EMAIL_HOST and SET.EMAIL_USER and SET.EMAIL_PASS),
    "gmail": _GAPI_OK and bool(SET.GMAIL_SA_JSON and SET.GMAIL_USER),
    "graph": bool(SET.MS_GRAPH_TOKEN),
    "gcal": _GAPI_OK and bool(SET.GCAL_SA_JSON),
    "places": bool(SET.GOOGLE_API_KEY),
    "webrtc": _WEBRTC_OK,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("agentic")


# ---------------- Persistence ----------------
def db_conn():
    return sqlite3.connect(SET.DB_PATH, check_same_thread=False)

def db_init():
    con = db_conn(); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, role TEXT, content TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS memories(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, kind TEXT, data TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS tasks(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, user_ask TEXT, plan TEXT, result TEXT
    )""")
    con.commit(); con.close()

db_init()


# ---------------- LLM helpers ----------------
def _openai_client() -> Optional["OpenAI"]:
    if not FEAT["openai"]:
        return None
    try:
        return OpenAI(api_key=SET.OPENAI_API_KEY)
    except Exception as e:
        log.error(f"OpenAI client init failed: {e}")
        return None

def llm_chat(messages: List[Dict[str, str]], temperature: float = 0.3) -> str:
    cli = _openai_client()
    if not cli:
        return "LLM unavailable. Provide OPENAI_API_KEY."
    try:
        out = cli.chat.completions.create(model=SET.MODEL_CHAT, messages=messages, temperature=temperature)
        return out.choices[0].message.content or ""
    except Exception as e:
        return f"(LLM error) {e}"

def llm_summarize(text: str, max_words: int = 120) -> str:
    return llm_chat(
        [{"role": "system", "content": "Summarize crisply for memory. Plain text."},
         {"role": "user", "content": f"Summarize within {max_words} words:\n{text}"}],
        0.2
    )

def llm_tts(text: str) -> Optional[bytes]:
    cli = _openai_client()
    if not cli:
        return None
    try:
        audio = cli.audio.speech.create(model=SET.MODEL_TTS, voice="alloy", input=text)
        if hasattr(audio, "content"):
            return audio.content
        if isinstance(audio, (bytes, bytearray)):
            return bytes(audio)
        if isinstance(audio, dict) and "audio" in audio:
            return base64.b64decode(audio["audio"])
    except Exception as e:
        log.error(f"TTS failed: {e}")
    return None

def llm_stt(audio_bytes: bytes) -> str:
    cli = _openai_client()
    if not cli:
        return ""
    try:
        buf = io.BytesIO(audio_bytes); buf.name = "audio.wav"
        tr = cli.audio.transcriptions.create(model=SET.MODEL_STT, file=buf)
        return tr.text or ""
    except Exception as e:
        log.error(f"STT failed: {e}")
        return ""


# ---------------- Memory store ----------------
class MemoryStore:
    def __init__(self, path: str): self.path = path
    def add(self, kind: str, data: Dict[str, Any]):
        con = db_conn(); cur = con.cursor()
        cur.execute("INSERT INTO memories(ts, kind, data) VALUES(?,?,?)",
                    (int(time.time()), kind, json.dumps(data)))
        con.commit(); con.close()
    def list(self, kind: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        con = db_conn(); cur = con.cursor()
        if kind:
            cur.execute("SELECT ts, kind, data FROM memories WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, limit))
        else:
            cur.execute("SELECT ts, kind, data FROM memories ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall(); con.close()
        out = []
        for ts, k, d in rows:
            try: out.append({"ts": ts, "kind": k, "data": json.loads(d)})
            except Exception: out.append({"ts": ts, "kind": k, "data": {"raw": d}})
        return out

MEM = MemoryStore(SET.DB_PATH)


# ---------------- Tools ----------------
def stocks_lookup(ticker: str, period: str = "3mo") -> Dict[str, Any]:
    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval="1d")
    if hist.empty:
        raise ValueError("No data returned.")
    latest = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) > 1 else latest
    price = float(latest["Close"])
    pct = float(((latest["Close"] - prev["Close"]) / prev["Close"]) * 100) if prev["Close"] else 0.0
    return {"ticker": ticker.upper(), "price": price, "pct": pct, "history": hist.reset_index()}

def plot_history(df: pd.DataFrame, ticker: str):
    fig = px.line(df, x="Date", y="Close", title=f"{ticker} — Close Price")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=32, b=10))
    st.plotly_chart(fig, use_container_width=True)

def ddg_news(query: str, n: int = 5) -> List[Dict[str, str]]:
    with DDGS() as ddg:
        hits = ddg.news(query, max_results=max(8, n)) or []
    return [{"title": h.get("title", "(untitled)"), "url": h.get("url", "#")} for h in hits[:n]]

def news_headlines(topic: str, n: int = 5) -> List[Dict[str, str]]:
    if FEAT["newsapi"]:
        try:
            r = requests.get("https://newsapi.org/v2/everything", params={
                "q": topic, "pageSize": n * 3, "sortBy": "publishedAt",
                "language": "en", "apiKey": SET.NEWS_API_KEY
            }, timeout=12)
            r.raise_for_status()
            arts = r.json().get("articles", [])[:n]
            return [{"title": a.get("title", "(untitled)"), "url": a.get("url", "#")} for a in arts]
        except Exception as e:
            log.error(f"NewsAPI failed: {e}")
    return ddg_news(topic, n)

def places_search(query: str, location: Optional[str] = None, n: int = 6) -> List[Dict[str, str]]:
    if FEAT["places"]:
        try:
            url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            params = {"query": query, "key": SET.GOOGLE_API_KEY}
            if location: params["location"] = location
            r = requests.get(url, params=params, timeout=10); r.raise_for_status()
            results = r.json().get("results", [])[:n]
            return [{"title": r1.get("name", "(untitled)"),
                     "url": f"https://www.google.com/maps/place/?q=place_id:{r1.get('place_id')}",
                     "body": r1.get("formatted_address", "")} for r1 in results]
        except Exception as e:
            log.error(f"Places API failed: {e}")
    hits = []
    try:
        with DDGS() as ddg:
            hits = ddg.text(query, max_results=n) or []
    except Exception:
        pass
    out = []
    for h in hits[:n]:
        out.append({"title": h.get("title", "(untitled)"),
                    "url": h.get("href") or h.get("url") or "#",
                    "body": h.get("body", "")})
    return out

def imap_fetch(date_filter: Optional[dt.date], sender_like: Optional[str], limit: int = 8):
    import imaplib, email
    from email.header import decode_header
    if not FEAT["imap"]:
        raise RuntimeError("IMAP not configured.")
    M = imaplib.IMAP4_SSL(SET.EMAIL_HOST, int(SET.EMAIL_PORT)) if SET.EMAIL_SSL else imaplib.IMAP4(SET.EMAIL_HOST, int(SET.EMAIL_PORT))
    M.login(SET.EMAIL_USER, SET.EMAIL_PASS)
    M.select(SET.EMAIL_FOLDER)
    criteria = ["ALL"]
    if date_filter:
        criteria += ["SENTSINCE", date_filter.strftime("%d-%b-%Y")]
        criteria += ["SENTBEFORE", (date_filter + dt.timedelta(days=1)).strftime("%d-%b-%Y")]
    if sender_like:
        criteria += ["FROM", sender_like]
    typ, data = M.search(None, *criteria)
    ids = data[0].split() if data and data[0] else []
    ids = ids[-limit:] if ids else []
    items = []
    for msgid in reversed(ids):
        typ, d2 = M.fetch(msgid, "(RFC822)")
        msg = email.message_from_bytes(d2[0][1])
        subj, enc = decode_header(msg.get("Subject", ""))[0]
        if isinstance(subj, bytes): subj = subj.decode(enc or "utf-8", errors="ignore")
        items.append({"from": msg.get("From", "(unknown)"),
                      "subject": subj or "(no subject)",
                      "date": msg.get("Date", "")})
    M.close(); M.logout()
    return items

def gmail_list_latest(max_results: int = 8) -> List[Dict[str, str]]:
    if not FEAT["gmail"]:
        raise RuntimeError("Gmail API not configured.")
    creds = SA_Credentials.from_service_account_info(json.loads(SET.GMAIL_SA_JSON))
    delegated = creds.with_subject(SET.GMAIL_USER)
    svc = gbuild("gmail", "v1", credentials=delegated)
    msgs = svc.users().messages().list(userId="me", maxResults=max_results).execute().get("messages", [])
    out = []
    for m in msgs:
        full = svc.users().messages().get(userId="me", id=m["id"], format="metadata",
                                          metadataHeaders=["From", "Subject", "Date"]).execute()
        headers = {h["name"]: h["value"] for h in full.get("payload", {}).get("headers", [])}
        out.append({"from": headers.get("From", ""),
                    "subject": headers.get("Subject", ""),
                    "date": headers.get("Date", "")})
    return out

def graph_list_latest(max_results: int = 8) -> List[Dict[str, str]]:
    if not FEAT["graph"]:
        raise RuntimeError("Microsoft Graph not configured.")
    headers = {"Authorization": f"Bearer {SET.MS_GRAPH_TOKEN}"}
    r = requests.get("https://graph.microsoft.com/v1.0/me/messages?$top="+str(max_results), headers=headers, timeout=10)
    r.raise_for_status()
    js = r.json().get("value", [])
    return [{"from": it.get("from", {}).get("emailAddress", {}).get("address", ""),
             "subject": it.get("subject", ""),
             "date": it.get("receivedDateTime", "")} for it in js]


# ---------------- Intent & planner ----------------
def parse_date_token(s: str) -> Optional[dt.date]:
    s = s.strip().lower()
    if "today" in s: return dt.date.today()
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m: return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    try: return dt.datetime.strptime(s, "%b %d %Y").date()
    except Exception: return None

def simple_intent(text: str) -> Dict[str, Any]:
    s = text.strip()
    m = re.search(r"(?:lookup|price|stock)\s+([A-Za-z\.\-]{1,10})(?:\s+(\d+[dwmy]|1y|2y|5y|max))?", s, re.I)
    if m: return {"type": "stock", "ticker": m.group(1).upper(), "period": (m.group(2) or "3mo").lower()}
    m = re.search(r"news(?:\s+about)?\s+(.+)", s, re.I)
    if m: return {"type": "news", "topic": m.group(1).strip()}
    if re.search(r"\b(email|emails|inbox)\b", s, re.I):
        md = re.search(r"from\s+([A-Za-z0-9._%+\-@]+)", s, re.I)
        sender = md.group(1) if md else None
        mdate = re.search(r"(today|\d{4}-\d{1,2}-\d{1,2}|[A-Za-z]{3}\s+\d{1,2}\s+\d{4})", s, re.I)
        date_ = parse_date_token(mdate.group(1)) if mdate else None
        return {"type": "email", "date": date_, "sender": sender}
    if re.search(r"\b(calendar|schedule|plan|agenda)\b", s, re.I):
        return {"type": "calendar", "text": s}
    if re.search(r"\b(near|in)\b", s, re.I) and re.search(r"(restaurant|cafe|hotel|museum|park|bar|food|breakfast|dinner|lunch)", s, re.I):
        return {"type": "places", "query": s}
    return {"type": "chat", "text": s}

def safe_call(fn, *a, **kw):
    try:
        return fn(*a, **kw), None
    except Exception as e:
        logging.exception("safe_call failure")
        return None, str(e)

def planner_execute(user_text: str) -> str:
    tools = {
        "stocks.lookup": lambda args: stocks_lookup(args.get("ticker", "AAPL"), args.get("period", "3mo")),
        "news.search":   lambda args: news_headlines(args.get("topic", "markets"), 3),
        "places.search": lambda args: places_search(args.get("query", "coffee near me"), None, 6),
        "calendar.list": lambda args: [],  # no-op unless gcal configured
        "email.imap":    lambda args: imap_fetch(args.get("date"), args.get("sender"), 8),
        "email.gmail":   lambda args: gmail_list_latest(8),
        "email.graph":   lambda args: graph_list_latest(8),
    }
    sysmsg = {"role": "system", "content": "You are a concise planner. Output JSON with {steps:[{tool,args,note}]}. Max 5 steps. No prose."}
    user = {"role": "user", "content": user_text}
    plan_json = llm_chat([sysmsg, user], temperature=0.2)
    try:
        plan = json.loads(plan_json).get("steps", [])
    except Exception:
        if re.search(r"tesla", user_text, re.I):
            plan = [
                {"tool": "stocks.lookup", "args": {"ticker": "TSLA", "period": "6mo"}, "note": "get price"},
                {"tool": "news.search",   "args": {"topic": "Tesla"}, "note": "headlines"},
            ]
        else:
            plan = [{"tool": "news.search", "args": {"topic": user_text}, "note": "context"}]

    results = []
    for i, step in enumerate(plan[:5]):
        tool = step.get("tool", ""); args = step.get("args", {})
        fn = tools.get(tool)
        if not fn:
            results.append({"step": i+1, "tool": tool, "error": "Unknown tool"})
            continue
        out, err = safe_call(fn, args)
        if err:
            results.append({"step": i+1, "tool": tool, "error": err})
        else:
            compact = out
            if isinstance(out, dict) and "history" in out:
                compact = dict(out); compact["history"] = f"[{len(out['history'])} rows]"  # why: avoid dumping large frames in memory
            results.append({"step": i+1, "tool": tool, "ok": True, "data": compact})

    final_text = llm_chat(
        [{"role": "system", "content": "Summarize tool results. Direct, six sentences max."},
         {"role": "user", "content": json.dumps(results)[:8000]}],
        0.2
    )

    con = db_conn(); cur = con.cursor()
    cur.execute("INSERT INTO tasks(ts, user_ask, plan, result) VALUES(?,?,?,?)",
                (int(time.time()), user_text, json.dumps(plan), json.dumps(results)))
    con.commit(); con.close()

    MEM.add("short", {"ask": user_text, "summary": llm_summarize(final_text)})
    return final_text


# ---------------- UI ----------------
st.set_page_config(page_title="Agentic AI", page_icon="🧠", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.0rem; padding-bottom: 1.2rem;}
  .assistant {background:#11141b;border:1px solid #262a33;border-radius:10px;padding:.75rem 1rem;}
  .pill {display:inline-block;padding:.35rem .55rem;border:1px solid #303645;border-radius:999px;font-size:.8rem;opacity:.85;margin-right:.25rem}
  .hint {background:#152132;border:1px solid #213147;border-radius:8px;padding:.7rem 1rem;margin:.4rem 0}
  .ticker {position:fixed;left:14px;bottom:14px;width:420px;background:#0f141c;border:1px solid #2a3341;border-radius:10px;padding:8px 10px;z-index:50;}
  .ticker .t {font-weight:600;opacity:.8;margin-bottom:6px}
  .ticker a {color:#e6ebf5;text-decoration:none}
  .ticker a:hover {text-decoration:underline}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Capabilities")
    st.caption(f"OpenAI: {'on' if FEAT['openai'] else 'off'} · News: {'on' if FEAT['newsapi'] else 'fallback'} · Places: {'on' if FEAT['places'] else 'fallback'}")
    st.caption(f"IMAP:{'on' if FEAT['imap'] else 'off'} · Gmail:{'on' if FEAT['gmail'] else 'off'} · Graph:{'on' if FEAT['graph'] else 'off'} · GCal:{'on' if FEAT['gcal'] else 'off'}")
    st.divider()
    t = st.text_input("Ticker", "AAPL")
    p = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=1)
    if st.button("Analyze", use_container_width=True):
        try:
            info = stocks_lookup(t, p)
            k1, k2 = st.columns(2)
            k1.metric("Price", f"${info['price']:.2f}")
            k2.metric("Change", f"{info['pct']:.2f}%")
            plot_history(info["history"], info["ticker"])
            for h in news_headlines(info["ticker"], 3):
                st.markdown(f"- [{h['title']}]({h['url']})")
        except Exception as e:
            st.error(f"Error: {e}")

c1, c2 = st.columns([1, 3])
with c1:
    st.markdown("### 🧠 Agentic AI")
with c2:
    st.markdown("<div class='hint'>Ask me to: <span class='pill'>lookup TSLA 6mo</span> <span class='pill'>news about energy</span> <span class='pill'>read my emails</span> <span class='pill'>schedule a 30-min sync tomorrow 2pm</span> <span class='pill'>best sushi in Seattle</span></div>", unsafe_allow_html=True)

voice_col1, voice_col2 = st.columns(2)
with voice_col1:
    enable_tts = st.toggle("🔊 Speak responses", value=False)
with voice_col2:
    enable_stt = st.toggle("🎙️ Microphone", value=False)

uploaded_audio = None
if enable_stt:
    if _WEBRTC_OK:
        st.info("Live mic via WebRTC enabled. Stop stream to process.")
        webrtc_streamer(key="mic", mode=WebRtcMode.SENDONLY, audio_receiver_size=1024)
    audio_file = st.file_uploader("Drop a short WAV/MP3 to transcribe", type=["wav","mp3","m4a"])
    if audio_file:
        uploaded_audio = audio_file.read()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ready. What do you need?"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(f"<div class='assistant'>{m['content']}</div>" if m["role"] == "assistant" else m["content"], unsafe_allow_html=True)

prompt = None
if uploaded_audio and FEAT["openai"]:
    st.info("Transcribing audio...")
    transcript = llm_stt(uploaded_audio)
    if transcript:
        prompt = transcript
prompt = st.chat_input("Ask for stocks, news, places, email, calendar…") if not prompt else prompt

def say(text: str):
    with st.chat_message("assistant"):
        st.markdown(f"<div class='assistant'>{text}</div>", unsafe_allow_html=True)
        if enable_tts and FEAT["openai"]:
            audio = llm_tts(text)
            if audio:
                b64 = base64.b64encode(audio).decode("ascii")
                st.audio(f"data:audio/mp3;base64,{b64}")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    intent = simple_intent(prompt)

    if intent["type"] == "stock":
        try:
            data = stocks_lookup(intent["ticker"], intent["period"])
            say(f"**{data['ticker']}** ${data['price']:.2f} ({data['pct']:.2f}%). Chart + headlines below.")
            plot_history(data["history"], data["ticker"])
            for h in news_headlines(data["ticker"], 3):
                st.markdown(f"- [{h['title']}]({h['url']})")
        except Exception as e:
            say(f"Error: {e}")

    elif intent["type"] == "news":
        hs = news_headlines(intent["topic"], 4)
        if not hs:
            say("No news found.")
        else:
            say(f"Latest on **{intent['topic']}**:")
            for h in hs:
                st.markdown(f"- [{h['title']}]({h['url']})")

    elif intent["type"] == "email":
        try:
            if FEAT["gmail"]:
                items = gmail_list_latest(8)
            elif FEAT["graph"]:
                items = graph_list_latest(8)
            elif FEAT["imap"]:
                items = imap_fetch(intent["date"], intent["sender"], 8)
            else:
                raise RuntimeError("No email backends configured.")
            title = "**Inbox — Today**" if not intent.get("date") else f"**Inbox — {intent['date'].isoformat()}**"
            say(title)
            if not items: st.caption("No matching emails.")
            for i, msg in enumerate(items):
                with st.expander(f"{i+1}. {msg.get('subject','(no subject)')}"):
                    st.write(f"From: {msg.get('from','')}")
                    st.write(f"Date: {msg.get('date','')}")
                    st.text_area("Reply", key=f"reply_{i}", placeholder="Type your reply…")
                    st.button("Send (disabled demo)", key=f"send_{i}")
            st.caption("Replies are disabled here. Provide OAuth + scopes to enable sending.")
        except Exception as e:
            say(f"Error: {e}")

    elif intent["type"] == "calendar":
        try:
            if FEAT["gcal"]:
                say("Upcoming events:")
                events = []  # plug actual calendar_list_events if you connect GCal
                if not events: st.caption("No upcoming events.")
            else:
                say("Calendar not connected. Add GCAL_SA_JSON to enable.")
        except Exception as e:
            say(f"Error: {e}")

    elif intent["type"] == "places":
        say("Here are a few spots I found:")
        hits = places_search(intent["query"], None, 6)
        if not hits: st.caption("No results found.")
        for h in hits[:6]:
            st.markdown(f"- **[{h['title']}]({h['url']})**  \n  <small>{h.get('body','')}</small>", unsafe_allow_html=True)

    else:
        if FEAT["openai"]:
            answer = planner_execute(prompt)
            say(answer)
        else:
            say("LLM unavailable. Provide OPENAI_API_KEY.")

# Sticky ticker
headlines = news_headlines("markets", 3) or [{"title": "Add NEWS_API_KEY for live headlines.", "url": "#"}]
st.markdown("<div class='ticker'>", unsafe_allow_html=True)
st.markdown("<div class='t'>🗞️ Latest Headlines</div>", unsafe_allow_html=True)
for h in headlines:
    st.markdown(f"• <a href='{h['url']}' target='_blank'>{h['title']}</a>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Memory
with st.expander("Memory"):
    mems = MEM.list(limit=10)
    if not mems:
        st.caption("No memories yet.")
    else:
        for m in mems:
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m["ts"]))
            st.markdown(f"- `{ts}` [{m['kind']}] {json.dumps(m['data'])[:160]}")

# Tasks
with st.expander("Tasks"):
    con = db_conn(); cur = con.cursor()
    cur.execute("SELECT ts, user_ask, plan FROM tasks ORDER BY id DESC LIMIT 10")
    rows = cur.fetchall(); con.close()
    if not rows:
        st.caption("No tasks yet.")
    else:
        for ts, ask, plan in rows:
            tstr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
            st.markdown(f"- `{tstr}` **{ask}**  \n  <small>{plan}</small>", unsafe_allow_html=True)
