import os
import time
import json
import threading
import traceback
import psycopg2
from datetime import datetime
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Optional PDF/DOCX support
try:
    from PyPDF2 import PdfReader
except:
    PdfReader = None
try:
    import docx
except:
    docx = None

# ---------------------------
# CONFIG
# ---------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID", "7561048693"))
LIVE_SUPPORT_LINK = "https://t.me/Justiceonsolana1"
SUPPORT_TIMEOUT = 120
DB_URL = os.getenv("DATABASE_URL")
PORT = int(os.getenv("PORT", 5000))

# ---------------------------
# DATABASE CONNECTION
# ---------------------------
conn = psycopg2.connect(DB_URL)
conn.autocommit = True
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY, text TEXT, batch_id INTEGER, added_at TIMESTAMP DEFAULT now()
)''')
c.execute('''CREATE TABLE IF NOT EXISTS stats (
    key TEXT PRIMARY KEY, value TEXT
)''')

def stat_get(key, default="0"):
    c.execute("SELECT value FROM stats WHERE key=%s", (key,))
    row = c.fetchone()
    return row[0] if row else default

def stat_set(key, value):
    c.execute("INSERT INTO stats(key,value) VALUES (%s,%s) ON CONFLICT(key) DO UPDATE SET value=%s",
              (str(key), str(value), str(value)))

if stat_get("total_users", None) is None:
    stat_set("total_users", "0")
if stat_get("total_queries", None) is None:
    stat_set("total_queries", "0")

# ---------------------------
# BOT INITIALIZATION
# ---------------------------
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="HTML")
START_TIME = time.time()
user_in_support = {}
admin_waiting_for_upload = set()

# ---------------------------
# EMBEDDINGS & GENERATIVE MODEL
# ---------------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
gen_model_name = "mosaicml/mpt-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
model = AutoModelForCausalLM.from_pretrained(gen_model_name, device_map="auto", torch_dtype=torch.float16)

all_chunks = []
all_embeddings = []

def rebuild_embeddings():
    global all_chunks, all_embeddings
    c.execute("SELECT text FROM chunks ORDER BY id")
    all_chunks = [r[0] for r in c.fetchall()]
    if all_chunks:
        all_embeddings = embed_model.encode(all_chunks, convert_to_tensor=True)
    else:
        all_embeddings = []

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
# KNOWLEDGE BASE MANAGEMENT
# ---------------------------
def append_chunks_to_db(chunks):
    batch = next_batch_id()
    for t in chunks:
        c.execute("INSERT INTO chunks(text,batch_id) VALUES (%s,%s)", (t, batch))
    rebuild_embeddings()
    return batch

def delete_last_batch():
    c.execute("SELECT MAX(batch_id) FROM chunks")
    last = c.fetchone()[0]
    if not last:
        return 0
    c.execute("DELETE FROM chunks WHERE batch_id=%s", (last,))
    rebuild_embeddings()
    return last

# ---------------------------
# SEMANTIC & GENERATIVE ANSWERING
# ---------------------------
def semantic_answer(query, top_k=5):
    if not all_chunks:
        return "I don’t have any knowledge yet. Ask the admin to upload a script."

    q_embedding = embed_model.encode([query], convert_to_tensor=True)
    sims = cosine_similarity(q_embedding.cpu().numpy(), all_embeddings.cpu().numpy())[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    selected = [all_chunks[i] for i in top_idx if sims[i] > 0.05]

    if not selected:
        return "I couldn’t find anything relevant in my knowledge base."

    context = "\n".join(selected)
    prompt = f"Answer naturally and directly using the context below:\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

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
# SUPPORT TIMER
# ---------------------------
def support_timer(cid):
    time.sleep(SUPPORT_TIMEOUT)
    if cid in user_in_support:
        del user_in_support[cid]
        bot.send_message(cid, "💡 Support session ended. You are now chatting with AI again.")

# ---------------------------
# PROCESS FILE/TEXT
# ---------------------------
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
        bot.send_message(cid, f"✅ Added {len(chunks)} new chunks (batch {batch}). Memory expanded successfully.")
    except Exception as e:
        traceback.print_exc()
        bot.send_message(cid, f"Error updating knowledge: {e}")

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
    bot.send_message(cid, "👋 Hello! I’m JusticeAI, your upgraded offline AI assistant. Ask me anything!")

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
    bot.send_message(msg.chat.id, f"🧹 Deleted last uploaded script (batch {deleted_batch}). Knowledge base updated.")

@bot.message_handler(func=lambda m: True, content_types=['text','document'])
def all_msgs(msg):
    try:
        cid = msg.chat.id
        text = (msg.text or "").strip()

        if any(k in text.lower() for k in ["connect me to live support", "talk to human", "support"]):
            bot.send_message(cid, f"🔗 Connect to support: {LIVE_SUPPORT_LINK}")
            bot.send_message(cid, "Say 'continue' or wait 2 minutes to return to AI.")
            user_in_support[cid] = time.time()
            threading.Thread(target=support_timer, args=(cid,), daemon=True).start()
            return

        if text.lower() == "continue" and cid in user_in_support:
            del user_in_support[cid]
            bot.send_message(cid, "✅ Back to AI mode. How can I assist?")
            return

        if cid in user_in_support:
            bot.send_message(cid, "🕒 You are with live support. Say 'continue' to talk to AI again.")
            return

        if cid == ADMIN_ID:
            txt_lower = text.lower()
            if txt_lower == "🩺 bot health":
                uptime = int(time.time()-START_TIME)
                reply = f"🩺 Uptime: {uptime}s\nUsers: {stat_get('total_users')}\nQueries: {stat_get('total_queries')}\nKnowledge items: {len(all_chunks)}\nTime: {now_str()}"
                bot.send_message(cid, reply)
                return
            if txt_lower == "📘 update knowledge":
                admin_waiting_for_upload.add(cid)
                bot.send_message(cid, "📄 Send a .txt/.pdf/.docx file or paste text to add new knowledge.")
                return
            if txt_lower == "📊 stats":
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

    except Exception as e:
        traceback.print_exc()
        bot.send_message(msg.chat.id, "⚠️ Error processing your message.")

# ---------------------------
# RUN LOOP
# ---------------------------
print(f"🚀 JusticeAI (Deep) running on port {PORT}...")
while True:
    try:
        bot.polling(non_stop=True, timeout=60, long_polling_timeout=60)
    except Exception as e:
        print("Polling error:", e)
        time.sleep(5)
