import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
import time
import requests
import re
import os

# ----------------------- PAGE CONFIG -----------------------
st.set_page_config(page_title="Agentic AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem;}
    .topbar {
        display: grid;
        grid-template-columns: 1fr 2fr 1fr;
        align-items: center;
        gap: 1rem;
        background: #0e1117;
        padding: .8rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .dropup-btn {
        position: fixed; bottom: 15px; right: 20px; z-index: 200;
    }
    .dropup-panel {
        position: fixed; bottom: 55px; right: 20px;
        width: 380px; max-height: 50vh; overflow: auto;
        background: #0e1117; border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px; padding: 10px;
        box-shadow: 0 8px 26px rgba(0,0,0,.4);
    }
    .news-ticker {
        position: fixed; left: 20px; bottom: 15px;
        width: 480px; z-index: 200; background: #0e1117;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px; padding: 10px;
        box-shadow: 0 8px 26px rgba(0,0,0,.45);
    }
</style>
""", unsafe_allow_html=True)

# ----------------------- SESSION STATE -----------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "dropup_open" not in st.session_state:
    st.session_state.dropup_open = False

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ----------------------- HELPERS -----------------------
def parse_command(cmd):
    cmd = cmd.strip().lower()
    if cmd.startswith("lookup"):
        m = re.match(r"lookup\s+([A-Za-z]+)\s*(\d+[dwmy])?", cmd)
        if m:
            return {"type": "stock", "ticker": m.group(1).upper(), "period": m.group(2) or "3mo"}
    elif "email" in cmd:
        return {"type": "email"}
    elif "news" in cmd:
        topic = cmd.replace("news about", "").strip() or "markets"
        return {"type": "news", "topic": topic}
    return {"type": "unknown"}

def record_history(cmd):
    st.session_state.history.append({"t": int(time.time()), "cmd": cmd})

def stock_data(ticker, period="3mo"):
    t = yf.Ticker(ticker)
    hist = t.history(period=period)
    if hist.empty:
        return None
    latest = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) > 1 else latest
    return {
        "price": float(latest["Close"]),
        "change": float(((latest["Close"] - prev["Close"]) / prev["Close"]) * 100),
        "data": hist.reset_index()
    }

def fetch_news(n=3, rotate_index=0, topic="markets"):
    if not NEWS_API_KEY:
        return [{"title": "Add NEWS_API_KEY in Streamlit secrets to enable headlines.", "url": "#"}]
    url = "https://newsapi.org/v2/everything"
    params = {"q": topic, "pageSize": 12, "sortBy": "publishedAt", "language": "en", "apiKey": NEWS_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=10).json()
        articles = r.get("articles", [])[:12]
        if not articles: return [{"title": "No news found.", "url": "#"}]
        start = rotate_index % len(articles)
        return [{"title": a["title"], "url": a["url"]} for a in articles[start:start+n]]
    except Exception as e:
        return [{"title": f"Error fetching news: {e}", "url": "#"}]

def inbox_today():
    inbox = [
        {"from": "CEO <ceo@corp.com>", "subject": "Strategy update meeting invite", "time": "09:15"},
        {"from": "HR <hr@corp.com>", "subject": "Policy changes: remote work", "time": "11:40"},
        {"from": "TechOps <ops@corp.com>", "subject": "Server patch completed", "time": "14:22"},
    ]
    st.subheader("Inbox — Today")
    for i, msg in enumerate(inbox, 1):
        st.markdown(f"**{i}. {msg['subject']}**  \n<small style='opacity:.7'>{msg['from']} • {msg['time']}</small>", unsafe_allow_html=True)
        with st.expander("Reply / Calendar options"):
            st.text_area("Reply", placeholder="Type your reply...")
            st.button("Send Reply", key=f"send_{i}")
            st.button("Add to Calendar", key=f"cal_{i}")

# ----------------------- TOP BAR -----------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown("### 🧠 Agentic AI")
with col2:
    user_cmd = st.text_input("Search or Command", placeholder="Try: lookup NVDA 6mo or news about energy", label_visibility="collapsed")
with col3:
    colA, colB, colC = st.columns(3)
    mv = colA.button("Market")
    ib = colB.button("Inbox")
    dr = colC.button("Report")
run = st.button("Run 🚀")

# ----------------------- SIDEBAR -----------------------
st.sidebar.header("Stock Lookup")
ticker = st.sidebar.text_input("Ticker", "AAPL")
period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1)
if st.sidebar.button("Analyze"):
    record_history(f"lookup {ticker} {period}")
data = stock_data(ticker, period)
if data:
    c1, c2 = st.sidebar.columns(2)
    c1.metric("Price", f"${data['price']:.2f}")
    c2.metric("Change", f"{data['change']:.2f}%")
    fig = px.line(data["data"], x="Date", y="Close", title=f"{ticker} — Close Price")
    st.sidebar.plotly_chart(fig, use_container_width=True)

# ----------------------- COMMAND PARSING -----------------------
panel = "market"
if run and user_cmd.strip():
    record_history(user_cmd)
    parsed = parse_command(user_cmd)
    if parsed["type"] == "stock":
        panel = "market"
        info = stock_data(parsed["ticker"], parsed["period"])
        if info:
            st.subheader(f"{parsed['ticker']} ({parsed['period']}) — ${info['price']:.2f}")
            fig = px.line(info["data"], x="Date", y="Close")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data found for that ticker.")
    elif parsed["type"] == "email":
        panel = "inbox"
    elif parsed["type"] == "news":
        panel = "market"
        st.subheader(f"📰 News about {parsed['topic'].title()}")
        for n in fetch_news(topic=parsed["topic"]):
            st.markdown(f"- [{n['title']}]({n['url']})")
    else:
        st.info("Try something like: lookup TSLA 3mo or news about energy.")

if mv: panel = "market"
if ib: panel = "inbox"
if dr: panel = "report"

# ----------------------- MAIN PANELS -----------------------
if panel == "market":
    st.subheader("Market Overview & Sentiment Analysis ↪")
    st.caption("Type a command above to lookup stocks or get news.")
elif panel == "inbox":
    inbox_today()
elif panel == "report":
    st.subheader("Daily Report")
    st.caption("Your summarized report will appear here.")

# ----------------------- HISTORY DROPUP -----------------------
if st.button("📜 History", key="history_btn"):
    st.session_state.dropup_open = not st.session_state.dropup_open
if st.session_state.dropup_open:
    st.markdown("<div class='dropup-panel'><b>Search History</b><br>", unsafe_allow_html=True)
    for h in reversed(st.session_state.history[-20:]):
        ts = time.strftime("%H:%M:%S", time.localtime(h["t"]))
        st.markdown(f"`{ts}` — {h['cmd']}")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------- NEWS TICKER -----------------------
rotation = int(time.time() // 60)
news = fetch_news(n=3, rotate_index=rotation)
st.markdown("<div class='news-ticker'><b>🗞️ Latest Headlines</b><br>", unsafe_allow_html=True)
for n in news:
    st.markdown(f"<a href='{n['url']}' target='_blank'>{n['title']}</a><br>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
