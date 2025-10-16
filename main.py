
import os
import time
import json
import threading
import traceback
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

import psycopg2
import numpy as np
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# transformers imports used only when loading model lazily
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Optional PDF/DOCX support
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None
try:
    import docx
except Exception:
    docx = None

# ---------------------------
# Logging
# ---------------------------
LOGFILE = "bot.log"
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler = RotatingFileHandler(LOGFILE, maxBytes=2_000_000, backupCount=3)
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
logger = logging.getLogger("justiceai")

# ---------------------------
# CONFIG
# ---------------------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    logger.error("TELEGRAM_TOKEN not set in env")
ADMIN_ID = int(os.getenv("ADMIN_ID", "7561048693"))
LIVE_SUPPORT_LINK = "https://t.me/Justiceonsolana1"
SUPPORT_TIMEOUT = int(os.getenv("SUPPORT_TIMEOUT", 120))
DB_URL = os.getenv("DATABASE_URL")
PORT = int(os.getenv("PORT", 5000))

if not DB_URL:
    logger.error("DATABASE_URL not set in env")

# ---------------------------
# DATABASE CONNECTION (Postgres)
# ---------------------------
conn = psycopg2.connect(DB_URL)
conn.autocommit = True
c = conn.cursor()

# Create tables
c.execute('''CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    text TEXT,
    batch_id INTEGER,
    added_at TIMESTAMP DEFAULT now()
)''')
c.execute('''CREATE TABLE IF NOT EXISTS stats (
    key TEXT PRIMARY KEY,
    value TEXT
)''')

def stat_get(key, default="0"):
    c.execute("SELECT value FROM stats WHERE key=%s", (key,))
    row = c.fetchone()
    return row[0] if row else default

def stat_set(key, value):
    c.execute(
        "INSERT INTO stats(key,value) VALUES (%s,%s) "
        "ON CONFLICT(key) DO UPDATE SET value=%s",
        (str(key), str(value), str(value))
    )

# Initialize basic stats if absent
if stat_get("total_users", None) is None:
    stat_set("total_users", "0")
if stat_get("total_queries", None) is None:
    stat_set("total_queries", "0")

# ---------------------------
# BOT SETUP
# ---------------------------
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="HTML")
START_TIME = time.time()
user_in_support = {}
admin_waiting_for_upload = set()

# ---------------------------
# EMBEDDINGS (local, small)
# ---------------------------
logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast
all_chunks = []          # list[str]
all_embeddings = None    # numpy array shape (n,d) or None

def rebuild_embeddings():
    """Reload chunks from DB and compute embeddings (numpy)."""
    global all_chunks, all_embeddings
    logger.info("Rebuilding embeddings from DB...")
    c.execute("SELECT text FROM chunks ORDER BY id")
    rows = c.fetchall()
    all_chunks = [r[0] for r in rows]
    if all_chunks:
        # compute numpy embeddings (not torch tensors) to keep memory small
        emb = embed_model.encode(all_chunks, convert_to_tensor=False, show_progress_bar=False)
        all_embeddings = np.asarray(emb)
        logger.info("Embeddings rebuilt: %d chunks, emb shape %s", len(all_chunks), all_embeddings.shape)
    else:
        all_embeddings = None
        logger.info("No chunks to embed.")

# initial load
try:
    rebuild_embeddings()
except Exception:
    logger.exception("Failed to build initial embeddings")

# ---------------------------
# GENERATIVE MODEL (lazy)
# ---------------------------
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME", "google/flan-t5-small")
tokenizer = None
gen_model = None
model_lock = threading.Lock()

def load_gen_model():
    """Lazily load tokenizer and model. Thread-safe."""
    global tokenizer, gen_model
    with model_lock:
        if gen_model is not None and tokenizer is not None:
            return True
        try:
            logger.info("Loading generative model: %s (this may take time)...", GEN_MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME, use_fast=True)
            # device_map="auto" requires accelerate installed; may require enough memory
            gen_model = AutoModelForCausalLM.from_pretrained(
               GEN_MODEL_NAME,
               torch_dtype=torch.float32
            )
            # put model in eval mode
            gen_model.eval()
            logger.info("Generative model loaded successfully.")
            return True
        except Exception:
            logger.exception("Failed to load generative model")
            tokenizer = None
            gen_model = None
            return False

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
    """Split text into char-limited chunks preserving word boundaries."""
    words = text.split()
    chunks = []
    buf, cur = [], 0
    for w in words:
        if cur + len(w) + 1 > max_chars:
            chunks.append(" ".join(buf))
            buf = buf[-20:]
            cur = sum(len(x)+1 for x in buf)
        buf.append(w)
        cur += len(w) + 1
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
# KNOWLEDGE PARSING (User/Bot dialogues)
# ---------------------------
def parse_dialogue_format(text):
    """
    Parse a dialogue text in lines containing 'User:' and 'Bot:'.
    For each Bot: line, pair it with the most recent preceding User: line (if any).
    Create a chunk '<user sentence> <bot reply>' (without labels).
    If Bot: appears with no preceding User:, store bot reply alone.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    responses = []
    last_user = None
    for ln in lines:
        low = ln.lower()
        if low.startswith("user:"):
            last_user = ln.split(":", 1)[1].strip()
        elif low.startswith("bot:"):
            bot_reply = ln.split(":", 1)[1].strip()
            if last_user:
                combined = f"{last_user}\n{bot_reply}"
                last_user = None
            else:
                combined = bot_reply
            responses.append(combined)
        else:
            # If line doesn't have explicit prefix, try to heuristically treat alternating lines:
            # if previous was user-like (question mark or short phrase), assume it's a user, else bot.
            if ln.endswith("?") or ln.lower().startswith(("hi","hello","hey","what","who","how","why","when")):
                last_user = ln
            else:
                # treat as bot reply
                if last_user:
                    combined = f"{last_user}\n{ln}"
                    last_user = None
                else:
                    combined = ln
                responses.append(combined)
    return responses

# ---------------------------
# DB helpers
# ---------------------------
def append_chunks_to_db(chunks):
    batch = next_batch_id()
    for t in chunks:
        c.execute("INSERT INTO chunks(text,batch_id) VALUES (%s,%s)", (t, batch))
    # Rebuild embeddings after inserting
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
# SEMANTIC + GENERATIVE ANSWERING
# ---------------------------
def semantic_answer(query, top_k=5):
    """Return a natural answer. Uses LLM if available; otherwise returns top chunks."""
    if not all_chunks or all_embeddings is None:
        return "I don’t have any knowledge yet. Ask the admin to upload a dialogue or text."

    try:
        # compute query embedding
        q_emb = embed_model.encode([query], convert_to_tensor=False, show_progress_bar=False)
        q_emb = np.asarray(q_emb)[0:1]  # shape (1,d)
        sims = cosine_similarity(q_emb, all_embeddings)[0]  # shape (n,)
        top_idx = sims.argsort()[-top_k:][::-1]
        selected = [all_chunks[i] for i in top_idx if sims[i] > 0.04]  # threshold slightly lower
    except Exception:
        logger.exception("Error during semantic retrieval")
        return "Internal retrieval error."

    if not selected:
        return "I couldn’t find anything relevant in my knowledge base."

    # If model not loaded, attempt to load (lazy)
    if gen_model is None or tokenizer is None:
        loaded = load_gen_model()
        if not loaded:
            # fallback: return top chunks directly (concise)
            logger.warning("Generative model unavailable; returning top chunks as fallback.")
            return "\n\n".join(selected[:3])

    # Prepare prompt: instruct model to answer naturally and not echo labels
    context = "\n\n".join(selected[:6])  # feed up to 6 chunks
    prompt = (
        "You are JusticeAI, a concise and helpful assistant. "
        "Answer the user's question naturally and directly using ONLY the context provided. "
        "Do NOT repeat the question or include any 'User:' or 'Bot:' labels. If the answer is not "
        "fully contained in the context, say you don't know and offer to look it up.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:"
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(next(gen_model.parameters()).device)
        outputs = gen_model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.95, temperature=0.2)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # The model sometimes echoes the prompt; try to remove prompt prefix if present
        if prompt.strip() in text:
            answer = text.split(prompt, 1)[-1].strip()
        else:
            # if model returns the full prompt+answer, we keep the trailing portion after "ANSWER:"
            if "ANSWER:" in text:
                answer = text.split("ANSWER:", 1)[-1].strip()
            else:
                answer = text.strip()
        # Safety: limit length
        if len(answer) > 4000:
            answer = answer[:4000] + "..."
        return answer
    except Exception:
        logger.exception("Generation failed; falling back to top chunks")
        return "\n\n".join(selected[:3])

# ---------------------------
# ADMIN / SUPPORT UI helpers
# ---------------------------
def admin_keyboard():
    kb = ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add(KeyboardButton("🩺 Bot Health"))
    kb.add(KeyboardButton("📘 Update Knowledge"))
    kb.add(KeyboardButton("📊 Stats"))
    return kb

def support_timer(cid):
    time.sleep(SUPPORT_TIMEOUT)
    if cid in user_in_support:
        del user_in_support[cid]
        try:
            bot.send_message(cid, "💡 Support session ended. You are now chatting with AI again.")
        except Exception:
            logger.exception("Failed to notify user about support timeout")

# ---------------------------
# FILE / TEXT PROCESSING
# ---------------------------
def process_file(path, cid):
    try:
        if path.lower().endswith(".pdf"):
            text = extract_text_from_pdf(path)
        elif path.lower().endswith(".docx"):
            text = extract_text_from_docx(path)
        else:
            text = open(path, "r", encoding="utf-8", errors="ignore").read()
        process_text(text, cid)
    except Exception:
        logger.exception("Error processing file")
        bot.send_message(cid, "⚠️ Error reading uploaded file.")
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

def process_text(text, cid):
    try:
        # Parse dialogue "User:" / "Bot:" into combined chunks (user+bot or bot alone)
        parsed = parse_dialogue_format(text)
        if not parsed:
            bot.send_message(cid, "⚠️ No valid Bot responses found in the file/text. Use 'User:' and 'Bot:' lines.")
            return
        chunks = []
        for r in parsed:
            chunks.extend(chunk_text(r))
        batch = append_chunks_to_db(chunks)
        bot.send_message(cid, f"✅ Added {len(chunks)} new chunks (batch {batch}). Knowledge updated.")
    except Exception:
        logger.exception("Error updating knowledge")
        bot.send_message(cid, "⚠️ Error updating knowledge.")

# ---------------------------
# BOT COMMANDS / HANDLERS
# ---------------------------
@bot.message_handler(commands=['start'])
def start_cmd(msg):
    cid = msg.chat.id
    seen = json.loads(stat_get("seen_users", "{}"))
    if str(cid) not in seen:
        seen[str(cid)] = now_str()
        stat_set("seen_users", json.dumps(seen))
        stat_set("total_users", str(int(stat_get("total_users", "0")) + 1))
    bot.send_message(cid, "👋 Hello! I’m JusticeAI. Ask me anything — I'll answer using my knowledge base.")

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
    bot.send_message(msg.chat.id, f"🧹 Deleted last uploaded script (batch {deleted_batch}). Knowledge updated.")

@bot.message_handler(commands=['logs'])
def send_logs(msg):
    if msg.chat.id != ADMIN_ID:
        bot.send_message(msg.chat.id, "❌ Unauthorized.")
        return
    try:
        with open(LOGFILE, "r", encoding="utf-8", errors="ignore") as f:
            data = f.read()[-3500:]
        bot.send_message(msg.chat.id, f"<pre>{data}</pre>", parse_mode="HTML")
    except Exception:
        logger.exception("Failed to read logs")
        bot.send_message(msg.chat.id, "⚠️ Could not read logs.")

@bot.message_handler(func=lambda m: True, content_types=['text','document'])
def all_msgs(msg):
    try:
        cid = msg.chat.id
        text = (msg.text or "").strip()

        # Live support request
        if any(k in text.lower() for k in ["connect me to live support", "talk to human", "support"]):
            bot.send_message(cid, f"🔗 Connect to support: {LIVE_SUPPORT_LINK}")
            bot.send_message(cid, "Say 'continue' or wait for the support timeout to return to AI.")
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

        # Admin-specific text commands
        if cid == ADMIN_ID:
            txt_lower = text.lower()
            if txt_lower == "🩺 bot health":
                uptime = int(time.time() - START_TIME)
                reply = (
                    f"🩺 Uptime: {uptime}s\n"
                    f"Users: {stat_get('total_users')}\n"
                    f"Queries: {stat_get('total_queries')}\n"
                    f"Knowledge items: {len(all_chunks)}\n"
                    f"Time: {now_str()}"
                )
                bot.send_message(cid, reply)
                return
            if txt_lower == "📘 update knowledge":
                admin_waiting_for_upload.add(cid)
                bot.send_message(cid, "📄 Send a .txt/.pdf/.docx file or paste dialogue text to add knowledge.")
                return
            if txt_lower == "📊 stats":
                total_users = stat_get("total_users")
                total_queries = stat_get("total_queries")
                seen = json.loads(stat_get("seen_users","{}"))
                last5 = list(seen.items())[-5:]
                txt = "\n".join([f"{u} @ {t}" for u,t in last5])
                bot.send_message(cid, f"📊 Stats:\nUsers: {total_users}\nQueries: {total_queries}\nLast 5:\n{txt}")
                return

        # Admin upload as plain text
        if cid in admin_waiting_for_upload and msg.content_type == 'text':
            admin_waiting_for_upload.discard(cid)
            threading.Thread(target=process_text, args=(msg.text,cid), daemon=True).start()
            return

        # Admin upload as document
        if cid == ADMIN_ID and msg.content_type == 'document' and cid in admin_waiting_for_upload:
            admin_waiting_for_upload.discard(cid)
            file_info = bot.get_file(msg.document.file_id)
            data = bot.download_file(file_info.file_path)
            fname = msg.document.file_name
            path = f"tmp_{int(time.time())}_{fname}"
            open(path, "wb").write(data)
            threading.Thread(target=process_file, args=(path,cid), daemon=True).start()
            return

        # Normal user query
        stat_set("total_queries", str(int(stat_get("total_queries", "0")) + 1))
        answer = semantic_answer(text)
        bot.send_message(cid, answer)

    except Exception:
        logger.exception("Error processing message")
        try:
            bot.send_message(msg.chat.id, "⚠️ Error processing your message.")
        except Exception:
            logger.exception("Failed to send error message to user")

# ---------------------------
# RUN LOOP
# ---------------------------
logger.info("🚀 JusticeAI (Deep, lazy-gen) starting. Port=%s", PORT)
while True:
    try:
        bot.polling(non_stop=True, timeout=60, long_polling_timeout=60)
    except Exception as e:
        logger.error("Polling error: %s", e, exc_info=True)
        time.sleep(5)
