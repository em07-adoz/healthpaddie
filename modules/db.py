# modules/db.py
import os
import json
from datetime import datetime

DB_FILE = "chat_history.json"

def save_chat_turn(user_text: str, bot_text: str, language: str):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "language": language,
        "user": user_text,
        "bot": bot_text
    }

    data = []
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []

    data.append(record)

    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
