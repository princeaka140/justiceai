import os
import time
import logging
import psycopg2
import torch
from telebot import TeleBot, types
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document

# =======================
# CONFIG & LOGGING
# =======================
load_dotenv()
logging.basicConfig(level=logging.INFO)

BOT_TOKEN = os.getenv("BOT_TOKEN")
DB_URL = os.getenv("DATABASE_URL")

bot = TeleBot(BOT_TOKEN)

# =======================
# DATABASE SETUP
# =======================
conn = psycopg2.connect(DB_URL)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS knowledge_base (
    id SERIAL PRIMARY KEY,
    question TEXT,
    answer TEXT,
    embedding BYTEA,
    created_at TIMESTAMP DEFAULT NOW()
)
""")
conn.commit()

# =======================
# MODEL LOAD (LIGHTWEIGHT)
# =======================
logging.info("Loading model: all-MiniLM-L6-v2")
device = "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# =======================
# BOT SETTINGS
# =======================
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
TYPING_ENABLED = True  # default ON

# =======================
# HELPER FUNCTIONS
# =======================

def simulate_typing(text: str):
    """Simulate typing delay based on text length."""
    if not TYPING_ENABLED:
        return
    delay = len(text) * 0.01  # 10ms per char
    time.sleep(min(delay, 5))  # cap at 5s max

def get_embedding(text):
    emb = model.encode(text, convert_to_tensor=True)
    return emb.cpu().numpy().tobytes()

def load_file_content(file_path):
    """Extract text from txt, pdf, or docx files."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return None

def fetch_best_answer(query):
    cursor.execute("SELECT id, question, answer, embedding FROM knowledge_base")
    rows = cursor.fetchall()
    if not rows:
        return None

    q_emb = model.encode(query, convert_to_tensor=True)
    best_score, best_answer = -1, None

    for _, q, a, emb_bytes in rows:
        emb_tensor = torch.tensor(torch.frombuffer(emb_bytes, dtype=torch.float32))
        sim = util.cos_sim(q_emb, emb_tensor).item()
        if sim > best_score:
            best_score, best_answer = sim, a

    if best_score > 0.55:  # threshold
        return best_answer
    return None

# =======================
# COMMAND HANDLERS
# =======================

@bot.message_handler(commands=["start"])
def start(msg):
    bot.send_message(msg.chat.id, "🤖 JusticeAI ready! Type your question anytime.")

@bot.message_handler(commands=["admin"])
def admin_menu(msg):
    if msg.from_user.id != ADMIN_ID:
        return bot.send_message(msg.chat.id, "⛔ Unauthorized")

    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton("🧠 Upload Knowledge", callback_data="upload"))
    markup.add(types.InlineKeyboardButton("🧹 Clear Database", callback_data="clear"))
    markup.add(types.InlineKeyboardButton("🔙 Forget Last Upload", callback_data="forget_last"))
    markup.add(types.InlineKeyboardButton("⏳ Toggle Typing", callback_data="toggle_typing"))
    markup.add(types.InlineKeyboardButton("📊 Stats", callback_data="stats"))
    markup.add(types.InlineKeyboardButton("💡 Health Check", callback_data="health"))
    bot.send_message(msg.chat.id, "⚙️ Admin Panel:", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: True)
def handle_admin_actions(call):
    global TYPING_ENABLED

    if call.data == "upload":
        bot.send_message(call.message.chat.id, "📤 Send your knowledge file or text now.")
        bot.register_next_step_handler(call.message, handle_knowledge_upload)

    elif call.data == "clear":
        cursor.execute("DELETE FROM knowledge_base")
        conn.commit()
        bot.send_message(call.message.chat.id, "🧹 Database cleared successfully.")

    elif call.data == "forget_last":
        cursor.execute("DELETE FROM knowledge_base WHERE id = (SELECT MAX(id) FROM knowledge_base)")
        conn.commit()
        bot.send_message(call.message.chat.id, "🗑️ Last entry removed.")

    elif call.data == "toggle_typing":
        TYPING_ENABLED = not TYPING_ENABLED
        bot.send_message(call.message.chat.id, f"⌛ Typing simulation {'enabled' if TYPING_ENABLED else 'disabled'}.")

    elif call.data == "stats":
        cursor.execute("SELECT COUNT(*) FROM knowledge_base")
        count = cursor.fetchone()[0]
        bot.send_message(call.message.chat.id, f"📚 Total entries: {count}")

    elif call.data == "health":
        bot.send_message(call.message.chat.id, "✅ JusticeAI is active and responsive.")

def handle_knowledge_upload(msg):
    """Handle text or file upload for new knowledge."""
    if msg.document:
        file_info = bot.get_file(msg.document.file_id)
        file_path = file_info.file_path
        downloaded = bot.download_file(file_path)
        tmp_path = "temp_upload" + os.path.splitext(file_info.file_path)[1]
        with open(tmp_path, "wb") as f:
            f.write(downloaded)
        content = load_file_content(tmp_path)
        os.remove(tmp_path)
    else:
        content = msg.text

    if not content:
        return bot.send_message(msg.chat.id, "⚠️ Unsupported file or empty content.")

    # Expect alternating Q/A format
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    for i in range(0, len(lines) - 1, 2):
        q, a = lines[i], lines[i + 1]
        emb = get_embedding(q)
        cursor.execute("INSERT INTO knowledge_base (question, answer, embedding) VALUES (%s,%s,%s)", (q, a, emb))
    conn.commit()

    bot.send_message(msg.chat.id, f"✅ Knowledge updated successfully! ({len(lines)//2} Q&A pairs)")

# =======================
# TEXT HANDLER
# =======================

@bot.message_handler(func=lambda msg: True, content_types=["text"])
def handle_question(msg):
    ans = fetch_best_answer(msg.text)
    if not ans:
        ans = "I don’t have that info yet."

    simulate_typing(ans)
    bot.send_message(msg.chat.id, ans)

# =======================
# RUN BOT
# =======================
logging.info("JusticeAI is running...")
bot.infinity_polling()
