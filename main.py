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
LIVE_SUPPORT_LINK = os.getenv("LIVE_SUPPORT_LINK", "https://t.me/Justiceonsolana1")
SUPPORT_TIMEOUT = int(os.getenv("SUPPORT_TIMEOUT", 120))
DB_URL = os.getenv("DATABASE_URL")
PORT = int(os.getenv("PORT", 5000))
# Optional: set to empty string to disable generation fallback
GEN_MODEL_NAME = os.getenv("GEN_MODEL_NAME", "google/flan-t5-small")

if not DB_URL:
    logger.error("DATABASE_URL not set in env")

# ---------------------------
# DATABASE CONNECTION (Postgres) - DO NOT CHANGE
# ---------------------------
conn = psycopg2.connect(DB_URL)
conn.autocommit = True
c = conn.cursor()

# Create tables (chunks stores text entries that we embed & search)
c.execute('''CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    kind TEXT,            -- 'user', 'bot', or 'pair'
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

# initialize simple stats
if stat_get("total_users", None) is None:
    stat_set("total_users", "0")
if stat_get("total_queries", None) is None:
    stat_set("total_queries", "0")
if stat_get("seen_users", None) is None:
    stat_set("seen_users", json.dumps({}))

# ---------------------------
# BOT SETUP
# ---------------------------
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode="HTML")
START_TIME = time.time()
user_in_support = {}
admin_waiting_for_upload = set()

# ---------------------------
# EMBEDDINGS (lightweight)
# ---------------------------
EMB_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "all-MiniLM-L6-v2")
logger.info("Loading embedding model (%s)...", EMB_MODEL_NAME)
embed_model = SentenceTransformer(EMB_MODEL_NAME, device="cpu")
# global index
all_chunks = []       # list[str] texts
all_kinds = []        # list[str] kinds aligned with all_chunks ('user'/'bot'/'pair')
all_ids = []          # list[int] DB ids aligned
all_embeddings = None # numpy array (n, d) float16

def rebuild_embeddings():
    """Reload chunks from DB and compute embeddings (numpy float16)."""
    global all_chunks, all_embeddings, all_kinds, all_ids
    logger.info("Rebuilding embeddings from DB...")
    c.execute("SELECT id, text, kind FROM chunks ORDER BY id")
    rows = c.fetchall()
    all_ids = [r[0] for r in rows]
    all_chunks = [r[1] for r in rows]
    all_kinds = [r[2] for r in rows]
    if all_chunks:
        # compute embeddings as float32 then cast to float16 to save memory
        emb = embed_model.encode(all_chunks, convert_to_tensor=False, show_progress_bar=False)
        all_embeddings = np.asarray(emb, dtype=np.float16)
        logger.info("Embeddings rebuilt: %d items, emb shape %s", len(all_chunks), all_embeddings.shape)
    else:
        all_embeddings = None
        logger.info("No chunks found in DB.")

try:
    rebuild_embeddings()
except Exception:
    logger.exception("Initial embedding rebuild failed")

# ---------------------------
# GENERATIVE MODEL (lazy; optional)
# ---------------------------
tokenizer = None
gen_model = None
model_lock = threading.Lock()

def load_gen_model():
    """Lazily load tokenizer & model if GEN_MODEL_NAME is set and available."""
    global tokenizer, gen_model
    if not GEN_MODEL_NAME:
        logger.info("GEN_MODEL_NAME empty -> generation disabled.")
        return False
    with model_lock:
        if gen_model is not None and tokenizer is not None:
            return True
        try:
            logger.info("Loading generative model: %s", GEN_MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME, use_fast=True)
            # small model recommended (flan-t5-small) to keep memory low; load to CPU
            gen_model = AutoModelForCausalLM.from_pretrained(GEN_MODEL_NAME, torch_dtype=torch.float32, low_cpu_mem_usage=True)
            gen_model.eval()
            logger.info("Generative model loaded.")
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
    words = text.split()
    chunks = []
    buf = []
    cur = 0
    for w in words:
        if cur + len(w) + 1 > max_chars:
            chunks.append(" ".join(buf))
            buf = buf[-20:]
            cur = sum(len(x) + 1 for x in buf)
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
# PARSING DIALOGUE FORMAT
# ---------------------------
def parse_dialogue_format(text):
    """
    Parse text containing 'User:' / 'Bot:' style lines.
    For each pair (user,b ot) produce three stored pieces:
      - 'user': the user line (helps paraphrase matching)
      - 'bot' : the bot reply (helps retrieval if user asks for a phrase from the reply)
      - 'pair': 'user\nbot' (keeps the combined QA context)
    If user/bot labels missing, heuristics attempt to alternate.
    Returns list of tuples (kind, text)
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    entries = []
    last_user = None
    for ln in lines:
        low = ln.lower()
        if low.startswith("user:"):
            last_user = ln.split(":", 1)[1].strip()
        elif low.startswith("bot:"):
            bot_reply = ln.split(":", 1)[1].strip()
            if last_user:
                entries.append(("user", last_user))
                entries.append(("bot", bot_reply))
                entries.append(("pair", last_user + "\n" + bot_reply))
                last_user = None
            else:
                # No preceding user: store bot alone
                entries.append(("bot", bot_reply))
        else:
            # heuristic: treat lines ending with ? or starting with question words as user
            if ln.endswith("?") or ln.lower().startswith(("who","what","how","why","where","when","do","did","is","are")):
                last_user = ln
            else:
                if last_user:
                    entries.append(("user", last_user))
                    entries.append(("bot", ln))
                    entries.append(("pair", last_user + "\n" + ln))
                    last_user = None
                else:
                    # orphan line -> treat as bot reply
                    entries.append(("bot", ln))
    return entries

# ---------------------------
# DB helpers (append/delete)
# ---------------------------
def append_chunks_to_db(chunks_with_kind):
    """
    chunks_with_kind: list of tuples (kind, text)
    inserts into DB and rebuilds embeddings
    """
    batch = next_batch_id()
    for kind, txt in chunks_with_kind:
        # store kind to help debugging/analysis
        c.execute("INSERT INTO chunks(text, kind, batch_id) VALUES (%s,%s,%s)", (txt, kind, batch))
    # rebuild index
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
# SEMANTIC RETRIEVAL + (optional) GENERATION
# ---------------------------
def semantic_answer(query, top_k=5):
    """
    1) Compute embedding for query
    2) Retrieve top_k chunks by cosine similarity
    3) If generative model loaded, ask it to produce a natural answer using top chunks as context
    4) Otherwise return the best bot chunk or combined pair.
    """
    if not all_chunks or all_embeddings is None:
        return "I don’t have any knowledge yet. Please ask the admin to upload a dialogue or text."

    try:
        q_emb = embed_model.encode([query], convert_to_tensor=False, show_progress_bar=False)
        q_emb = np.asarray(q_emb, dtype=np.float16)[0:1]  # shape (1,d)
        # use float32 for similarity calculation to avoid precision pitfalls
        sims = cosine_similarity(q_emb.astype(np.float32), all_embeddings.astype(np.float32))[0]
        top_idx = sims.argsort()[-top_k:][::-1]
        retrieved = [(i, sims[i], all_kinds[i], all_chunks[i]) for i in top_idx]
    except Exception:
        logger.exception("Retrieval error")
        return "Internal retrieval error."

    # filter by a reasonable threshold (helps avoid false positives on tiny KBs)
    retrieved = [r for r in retrieved if r[1] >= 0.30]  # tweakable
    if not retrieved:
        # if nothing above threshold, return best single item (helpful when KB tiny)
        best_idx = sims.argsort()[-1]
        return all_chunks[best_idx]

    # prefer 'pair' kind or 'bot' kind for direct answers
    # assemble context from top unique items (prioritize pair then bot then user)
    seen_texts = set()
    context_pieces = []
    for _, score, kind, txt in retrieved:
        if txt in seen_texts:
            continue
        seen_texts.add(txt)
        # prefer showing bot/pair content in context
        if kind == "pair":
            context_pieces.append(txt)
        elif kind == "bot":
            context_pieces.append(txt)
        else:
            # keep user pieces later
            context_pieces.append(txt)

    context = "\n\n".join(context_pieces[:4])

    # attempt generation only if model available or if explicitly configured
    if GEN_MODEL_NAME and (gen_model is None or tokenizer is None):
        ok = load_gen_model()
        if not ok:
            logger.info("Generation model not available; returning best retrieved chunk.")
            # return the best bot/pair if present
            for _,_,k,t in retrieved:
                if k in ("pair","bot"):
                    return t
            return retrieved[0][3]

    if gen_model is None or tokenizer is None:
        # generation disabled or failed -> return best matching bot/pair or pair text
        for _,_,k,t in retrieved:
            if k in ("pair","bot"):
                return t
        return retrieved[0][3]

    # Build a compact prompt instructing the model to answer naturally without labels.
    prompt = (
        "You are JusticeAI, a concise friendly assistant. Use ONLY the context below to answer naturally.\n\n"
        f"Context:\n{context}\n\nUser: {query}\nJusticeAI:"
    )

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(next(gen_model.parameters()).device)
        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.4,
                top_p=0.9,
            )
        out = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # strip any leading prompt echo
        if "JusticeAI:" in out:
            answer = out.split("JusticeAI:", 1)[-1].strip()
        elif "Answer:" in out:
            answer = out.split("Answer:", 1)[-1].strip()
        else:
            # model might return only the answer fragment
            answer = out.strip()
        # safety: keep answer reasonable length
        if len(answer) > 3000:
            answer = answer[:3000] + "..."
        return answer
    except Exception:
        logger.exception("Generation failed; fallback to retrieved.")
        for _,_,k,t in retrieved:
            if k in ("pair","bot"):
                return t
        return retrieved[0][3]

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
        parsed = parse_dialogue_format(text)
        if not parsed:
            bot.send_message(cid, "⚠️ No valid Bot responses found. Use lines like 'User: ...' and 'Bot: ...'.")
            return
        # parsed is list of (kind, text)
        batch = append_chunks_to_db(parsed)
        bot.send_message(cid, f"✅ Added {len(parsed)} new items (batch {batch}). Knowledge updated.")
    except Exception:
        logger.exception("Error updating knowledge")
        bot.send_message(cid, "⚠️ Error updating knowledge.")

# ---------------------------
# BOT COMMANDS / HANDLERS
# ---------------------------
@bot.message_handler(commands=['start'])
def start_cmd(msg):
    cid = msg.chat.id
    seen = json.loads(stat_get("seen_users", "{}") or "{}")
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

        # support
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

        # admin commands
        if cid == ADMIN_ID:
            tl = text.lower()
            if tl == "🩺 bot health":
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
            if tl == "📘 update knowledge":
                admin_waiting_for_upload.add(cid)
                bot.send_message(cid, "📄 Send a .txt/.pdf/.docx file or paste dialogue text to add knowledge.")
                return
            if tl == "📊 stats":
                total_users = stat_get("total_users")
                total_queries = stat_get("total_queries")
                seen = json.loads(stat_get("seen_users","{}") or "{}")
                last5 = list(seen.items())[-5:]
                txt = "\n".join([f"{u} @ {t}" for u,t in last5])
                bot.send_message(cid, f"📊 Stats:\nUsers: {total_users}\nQueries: {total_queries}\nLast 5:\n{txt}")
                return

        # admin upload as text
        if cid in admin_waiting_for_upload and msg.content_type == 'text':
            admin_waiting_for_upload.discard(cid)
            threading.Thread(target=process_text, args=(msg.text,cid), daemon=True).start()
            return

        # admin upload as document
        if cid == ADMIN_ID and msg.content_type == 'document' and cid in admin_waiting_for_upload:
            admin_waiting_for_upload.discard(cid)
            file_info = bot.get_file(msg.document.file_id)
            data = bot.download_file(file_info.file_path)
            fname = msg.document.file_name
            path = f"tmp_{int(time.time())}_{fname}"
            open(path, "wb").write(data)
            threading.Thread(target=process_file, args=(path,cid), daemon=True).start()
            return

        # normal user query
        stat_set("total_queries", str(int(stat_get("total_queries", "0")) + 1))
        ans = semantic_answer(text)
        bot.send_message(cid, ans)

    except Exception:
        logger.exception("Error processing message")
        try:
            bot.send_message(msg.chat.id, "⚠️ Error processing your message.")
        except Exception:
            logger.exception("Failed to send error message to user")

# ---------------------------
# RUN LOOP
# ---------------------------
logger.info("🚀 JusticeAI starting on port %s", PORT)
while True:
    try:
        bot.polling(non_stop=True, timeout=60, long_polling_timeout=60)
    except Exception as e:
        logger.error("Polling error: %s", e, exc_info=True)
        time.sleep(5)
