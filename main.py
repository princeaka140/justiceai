import os
import time
import json
import threading
import traceback
from datetime import datetime
from flask import Flask
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
import psycopg2

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None
try:
    import docx
except Exception:
    docx = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# CONFIG
# ---------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
ADMIN_ID = 7561048693
LIVE_SUPPORT_LINK = "https://t.me/Justiceonsolana1"
SUPPORT_TIMEOUT = 120

bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="HTML")
START_TIME = time.time()
user_in_support = {}
admin_waiting_for_upload = set()

# ---------------------------
# DATABASE (PostgreSQL)
# ---------------------------
conn = psycopg2.connect(DATABASE_URL, sslmode="require")
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    text TEXT,
    batch_id INTEGER,
    added_at TEXT
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS stats (
    key TEXT PRIMARY KEY,
    value TEXT
)
""")
conn.commit()

def stat_get(key, default="0"):
    c.execute("SELECT value FROM stats WHERE key=%s", (key,))
    row = c.fetchone()
    return row[0] if row else default

def stat_set(key, value):
    c.execute("""
        INSERT INTO stats (key, value)
        VALUES (%s, %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
    """, (key, str(value)))
    conn.commit()

if stat_get("total_users", None) is None:
    stat_set("total_users", "0")
if stat_get("total_queries", None) is None:
    stat_set("total_queries", "0")

# ---------------------------
# UTILITIES
# ---------------------------
def now_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def next_batch_id():
    c.execute("SELECT MAX(batch_id) FROM chunks")
    val = c.fetchone()[0]
    return (val or 0) + 1

def chunk_text(text, max_chars=700):
    words = text.split()
    chunks = []
    buf, cur = [], 0
    for w in words:
        if cur + len(w) + 1 > max_chars:
            chunks.append(" ".join(buf))
            buf = buf[-20:]
            cur = sum(len(x)+1 for x in buf)
        buf.append(w)
        cur += len(w)+1
    if buf:
        chunks.append(" ".join(buf))
    return chunks

def extract_text_from_pdf(path):
    if not PdfReader:
        raise RuntimeError("PyPDF2 not installed")
    reader = PdfReader(path)
    texts = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(texts)

def extract_text_from_docx(path):
    if not docx:
        raise RuntimeError("python-docx not installed")
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)

# ---------------------------
# KNOWLEDGE BASE
# ---------------------------
def append_chunks_to_db(chunks):
    batch = next_batch_id()
    for t in chunks:
        c.execute("INSERT INTO chunks(text,batch_id,added_at) VALUES (%s,%s,%s)", (t, batch, now_str()))
    conn.commit()
    return batch

def delete_last_batch():
    c.execute("SELECT MAX(batch_id) FROM chunks")
    last = c.fetchone()[0]
    if not last:
        return 0
    c.execute("DELETE FROM chunks WHERE batch_id=%s", (last,))
    conn.commit()
    return last

def get_all_chunks():
    c.execute("SELECT text FROM chunks")
    return [r[0] for r in c.fetchall()]

vectorizer = None
tfidf_matrix = None
all_chunks = []

def rebuild_vector_index():
    global vectorizer, tfidf_matrix, all_chunks
    all_chunks = get_all_chunks()
    if not all_chunks:
        vectorizer = None
        tfidf_matrix = None
        return
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_chunks)

def semantic_answer(query):
    global vectorizer, tfidf_matrix, all_chunks
    if vectorizer is None or tfidf_matrix is None:
        rebuild_vector_index()
    if vectorizer is None or not all_chunks:
        return "I don’t have any knowledge yet. Ask the admin to upload a script."

    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]
    top_idx = sims.argsort()[-5:][::-1]
    selected = [all_chunks[i] for i in top_idx if sims[i] > 0.05]

    if not selected:
        return "I couldn’t find anything relevant in my knowledge base."

    context = "\n".join(selected)
    return generate_wise_reply(context, query)

def generate_wise_reply(context, query):
    summary = summarize(context, query)
    reasoning = (f"🤔 Based on everything I’ve learned, here’s what best answers your question:\n\n{summary}\n\n💡 In essence, about '{query}', the key point is that {extract_key_point(summary)}")
    return reasoning

def summarize(context, query):
    sentences = context.split(".")
    matches = [s.strip() for s in sentences if any(w in s.lower() for w in query.lower().split())]
    summary = ". ".join(matches[:4]) or ". ".join(sentences[:3])
    return summary.strip()

def extract_key_point(text):
    words = text.split()
    if len(words) > 30:
        return " ".join(words[:30]) + "..."
    return text

# ---------------------------
# ADMIN DASHBOARD
# ---------------------------
def admin_keyboard():
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("🩺 Bot Health"))
    kb.add(KeyboardButton("📘 Update Knowledge"))
    kb.add(KeyboardButton("📊 Stats"))
    return kb

# ---------------------------
# BOT HANDLERS
# ---------------------------
@bot.message_handler(commands=['start'])
def start_cmd(msg):
    cid = msg.chat.id
    seen = json.loads(stat_get("seen_users", "{}"))
    if str(cid) not in seen:
        seen[str(cid)] = now_str()
        stat_set("seen_users", json.dumps(seen))
        total = int(stat_get("total_users", "0")) + 1
        stat_set("total_users", str(total))
    bot.send_message(cid, "👋 Hello! I’m your offline AI assistant. Ask me anything based on what I’ve learned.")

@bot.message_handler(commands=['dashboard'])
def dashboard(msg):
    if msg.chat.id != ADMIN_ID:
        bot.send_message(msg.chat.id, "❌ Unauthorized.")
        return
    bot.send_message(msg.chat.id, "🛠 Admin dashboard:", reply_markup=admin_keyboard())

@bot.message_handler(commands=['forget_last'])
def forget_last(msg):
    if msg.chat.id != ADMIN_ID:
        bot.send_message(msg.chat.id, "❌ Unauthorized.")
        return
    deleted_batch = delete_last_batch()
    if not deleted_batch:
        bot.send_message(msg.chat.id, "⚠️ No previous uploads found.")
        return
    rebuild_vector_index()
    bot.send_message(msg.chat.id, f"🧹 Deleted last uploaded script (batch {deleted_batch}). Knowledge base updated.")

@bot.message_handler(func=lambda m: True, content_types=['text','document'])
def all_msgs(msg):
    try:
        cid = msg.chat.id
        text = (msg.text or "").strip().lower()

        if any(k in text for k in ["connect me to live support", "talk to human", "support"]):
            bot.send_message(cid, f"🔗 Connect to support: {LIVE_SUPPORT_LINK}")
            bot.send_message(cid, "Say 'continue' or wait 2 minutes to return to AI.")
            user_in_support[cid] = time.time()
            threading.Thread(target=support_timer, args=(cid,), daemon=True).start()
            return

        if text == "continue" and cid in user_in_support:
            del user_in_support[cid]
            bot.send_message(cid, "✅ Back to AI mode. How can I assist?")
            return

        if cid in user_in_support:
            bot.send_message(cid, "🕒 You are with live support. Say 'continue' to talk to AI again.")
            return

        if cid == ADMIN_ID and msg.text:
            if text == "🩺 bot health":
                uptime = int(time.time()-START_TIME)
                reply = f"🩺 Uptime: {uptime}s\nUsers: {stat_get('total_users')}\nQueries: {stat_get('total_queries')}\nKnowledge items: {len(get_all_chunks())}\nTime: {now_str()}"
                bot.send_message(cid, reply)
                return
            if text == "📘 update knowledge":
                admin_waiting_for_upload.add(cid)
                bot.send_message(cid, "📄 Send a .txt/.pdf/.docx file or paste text to add new knowledge.")
                return
            if text == "📊 stats":
                total_users = stat_get("total_users")
                total_queries = stat_get("total_queries")
                seen = json.loads(stat_get("seen_users","{}"))
                last5 = list(seen.items())[-5:]
                txt = "\n".join([f"{u} @ {t}" for u,t in last5])
                bot.send_message(cid, f"📊 Stats:\nUsers: {total_users}\nQueries: {total_queries}\nLast 5:\n{txt}")
                return

        if cid in admin_waiting_for_upload and msg.content_type == 'text':
            admin_waiting_for_upload.discard(cid)
            threading.Thread(target=process_text, args=(msg.text,cid), daemon=True).start()
            return

        if cid == ADMIN_ID and msg.content_type == 'document' and cid in admin_waiting_for_upload:
            admin_waiting_for_upload.discard(cid)
            file_info = bot.get_file(msg.document.file_id)
            data = bot.download_file(file_info.file_path)
            fname = msg.document.file_name
            path = f"tmp_{int(time.time())}_{fname}"
            open(path,"wb").write(data)
            threading.Thread(target=process_file, args=(path,cid), daemon=True).start()
            return

        total_queries = int(stat_get("total_queries","0"))+1
        stat_set("total_queries",str(total_queries))
        ans = semantic_answer(msg.text)
        bot.send_message(cid, ans)
    except Exception:
        traceback.print_exc()
        bot.send_message(msg.chat.id, "⚠️ Error processing your message.")

# ---------------------------
# HELPERS
# ---------------------------
def support_timer(cid):
    time.sleep(SUPPORT_TIMEOUT)
    if cid in user_in_support:
        del user_in_support[cid]
        bot.send_message(cid, "💡 Support session ended. You are now chatting with AI again.")

def process_file(path, cid):
    try:
        if path.lower().endswith(".pdf"):
            text = extract_text_from_pdf(path)
        elif path.lower().endswith(".docx"):
            text = extract_text_from_docx(path)
        else:
            text = open(path,"r",encoding="utf-8",errors="ignore").read()
        process_text(text, cid)
    except Exception as e:
        traceback.print_exc()
        bot.send_message(cid, f"Error reading file: {e}")
    finally:
        try: os.remove(path)
        except: pass

def process_text(text, cid):
    try:
        chunks = chunk_text(text)
        batch = append_chunks_to_db(chunks)
        rebuild_vector_index()
        bot.send_message(cid, f"✅ Added {len(chunks)} new chunks (batch {batch}). Memory expanded successfully.")
    except Exception as e:
        traceback.print_exc()
        bot.send_message(cid, f"Error updating knowledge: {e}")

# ---------------------------
# FLASK WEB SERVER (PORT BIND)
# ---------------------------
app = Flask(__name__)

@app.route('/')
def home():
    return "JusticeAI bot is alive."

def run_flask():
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

threading.Thread(target=run_flask, daemon=True).start()

# ---------------------------
# RUN LOOP
# ---------------------------
print("🚀 JusticeAI bot is running on Render with PostgreSQL...")
while True:
    try:
        bot.polling(non_stop=True, timeout=60, long_polling_timeout=60)
    except Exception as e:
        print("Polling error:", e)
        time.sleep(5)
