# healthpaddie.py


import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from modules.utils import clear_screen
from modules.tts import text_to_speech
from modules.db import save_chat_turn
from utils.sound_config import play_audio

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# ----- Language configuration -----

LANGUAGE_OPTIONS = {
    "1": {
        "name": "English",
        "code": "en",
        "instruction": "Respond in clear, simple English."
    },
    "2": {
        "name": "Hausa",
        "code": "ha",
        "instruction": "Respond in Hausa, using clear and simple language."
    },
    "3": {
        "name": "Yoruba",
        "code": "yo",
        "instruction": "Respond in Yoruba, using clear and simple language."
    },
    "4": {
        "name": "Igbo",
        "code": "ig",
        "instruction": "Respond in Igbo, using clear and simple language."
    },
}

INTRO_FILE = "intro_message.txt"

def load_intro_message():
    if os.path.exists(INTRO_FILE):
        with open(INTRO_FILE, "r", encoding="utf-8") as f:
            return f.read()
    # Fallback text if file missing
    return """
🌿 Welcome to HealthPaddie
Your trusted companion for verified health information.

1 - English
2 - Hausa
3 - Yoruba
4 - Igbo
"""

# ----- Model & Vectorstore setup -----

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

VECTORSTORE_DIR = "./vectorstore"

if not os.path.exists(VECTORSTORE_DIR):
    raise ValueError(
        "Vectorstore not found. Run 'python build_vectorstore.py' first to create it."
    )

vectorstore = FAISS.load_local(
    VECTORSTORE_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant"
)

BASE_SYSTEM_PROMPT = """
You are HealthPaddie, a safe and trusted health information assistant.
You MUST:
- Answer ONLY using the information from the provided context (RAG documents).
- If the answer is not in the context, say you do not know and advise the user to consult a healthcare professional.
- Give short, clear explanations (2–4 short paragraphs).
- Use friendly and respectful language.
- This is not a medical diagnosis. Always remind users to consult a doctor for serious issues.

User language instruction:
{language_instruction}
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", BASE_SYSTEM_PROMPT),
        (
            "human",
            "Chat history:\n{history}\n\nUser question:\n{question}\n\nRelevant health information:\n{context}\n"
        ),
    ]
)

chat_history = []

# ----- Main functions -----

def select_language():
    clear_screen()
    intro = load_intro_message()
    print(intro)
    choice = None
    while choice not in LANGUAGE_OPTIONS:
        choice = input("Enter language number (1-4): ").strip()
        if choice not in LANGUAGE_OPTIONS:
            print("Invalid choice. Please select 1, 2, 3, or 4.")
    language = LANGUAGE_OPTIONS[choice]
    clear_screen()
    print(f"✅ You selected: {language['name']}\n")
    return language

def build_prompt(question: str, language_instruction: str):
    docs = retriever.invoke(question)
    context_text = "\n\n".join(d.page_content for d in docs)

    history_text = "\n".join(chat_history[-10:])  # last 10 turns

    prompt = PROMPT_TEMPLATE.format(
        language_instruction=language_instruction,
        history=history_text,
        question=question,
        context=context_text,
    )
    return prompt

def chat_loop(language):
    lang_name = language["name"]
    lang_code = language["code"]
    lang_instruction = language["instruction"]

    print(f"👋 You are now chatting with HealthPaddie in {lang_name}.")
    print("Type 'exit' or 'quit' to end the chat.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("HealthPaddie: Thank you for using HealthPaddie. Stay healthy! 💚")
            break

        if not user_input:
            continue

        prompt = build_prompt(user_input, lang_instruction)
        response = llm.invoke(prompt)

        bot_reply = response.content.strip()

        print(f"\nHealthPaddie ({lang_name}): {bot_reply}\n")

        # Save to in-memory history
        chat_history.append(f"User: {user_input}")
        chat_history.append(f"Bot: {bot_reply}")

        # Save to simple JSON "database"
        save_chat_turn(user_input, bot_reply, lang_name)

        # Optional TTS
        use_tts = input("🔊 Play audio? (y/n): ").strip().lower()
        if use_tts == "y":
            #speak_text(bot_reply, lang_code=lang_code)
            file_name = text_to_speech(bot_reply)
            play_audio(file_name)



def main():
    language = select_language()
    chat_loop(language)

if __name__ == "__main__":
    main()
