# streamlit_app.py

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from modules.tts import text_to_speech
from modules.db import save_chat_turn

# ---------- ENV + CONSTANTS ----------

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

LANGUAGE_OPTIONS = {
    "English": {
        "code": "en",
        "instruction": "Respond in clear, simple English."
    },
    "Hausa": {
        "code": "ha",
        "instruction": "Respond in Hausa, using clear and simple language."
    },
    "Yoruba": {
        "code": "yo",
        "instruction": "Respond in Yoruba, using clear and simple language."
    },
    "Igbo": {
        "code": "ig",
        "instruction": "Respond in Igbo, using clear and simple language."
    },
}

INTRO_TEXT = """
🌿 Welcome to HealthPaddie
Your trusted companion for verified health information.

Please select your language:
/ Zaɓi harshe / Yan ede / Họrọ asụsụ:

1 - 🇬🇧 English
    Welcome! I can answer your health questions using verified and trusted information.

2 - 🇳🇬 Hausa (هَوْسَ)
    Barka da zuwa! Zan ba ka amsa ga tambayoyin lafiya daga sahihan bayanai na gaskiya.

3 - 🇳🇬 Yoruba
    Kaabọ! Mo le dahun ibeere ilera rẹ nipa lilo alaye to dájú ati to pe.

4 - 🇳🇬 Igbo
    Nnọọ! Enwere m ike ịza ajụjụ gị gbasara ahụike site na ozi dị mma na e kwadoro.

Type the number of your language choice (1–4) to continue.
"""

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

# ---------- MODEL & VECTORSTORE SETUP ----------

@st.cache_resource
def load_vectorstore_and_llm():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore_dir = "./vectorstore"
    if not os.path.exists(vectorstore_dir):
        raise ValueError(
            "Vectorstore not found. Run 'python build_vectorstore.py' first to create it."
        )

    vectorstore = FAISS.load_local(
        vectorstore_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant",
    )

    return retriever, llm


retriever, llm = load_vectorstore_and_llm()

# ---------- HELPER FUNCTIONS ----------

def build_prompt(question: str, language_instruction: str, chat_history):
    docs = retriever.invoke(question)

    context_text = "\n\n".join(d.page_content for d in docs)
    history_text = "\n".join(chat_history[-10:])

    prompt = PROMPT_TEMPLATE.format(
        language_instruction=language_instruction,
        history=history_text,
        question=question,
        context=context_text,
    )
    return prompt


def run_tts_and_prepare_audio(bot_reply: str) -> bytes | None:
    """
    Use your existing Groq TTS to generate an MP3, then return the bytes
    for Streamlit to play with st.audio().
    """
    try:
        file_path = text_to_speech(bot_reply)  # returns path to mp3
        path = Path(file_path)
        if path.exists():
            return path.read_bytes()
    except Exception as e:
        st.warning(f"⚠ TTS error: {e}")
    return None


# ---------- STREAMLIT UI ----------

def main():
    st.set_page_config(page_title="HealthPaddie", page_icon="🌿", layout="centered")

    st.title("🌿 HealthPaddie")
    st.markdown(INTRO_TEXT)

    # Session state init
    if "chat_history" in st.session_state:
        chat_history = st.session_state["chat_history"]
    else:
        st.session_state["chat_history"] = []
        chat_history = []

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Sidebar – language + options
    st.sidebar.header("Settings")
    language_name = st.sidebar.selectbox(
        "Language",
        list(LANGUAGE_OPTIONS.keys()),
        index=0,
    )
    language = LANGUAGE_OPTIONS[language_name]
    lang_instruction = language["instruction"]
    lang_code = language["code"]

    enable_tts = st.sidebar.checkbox("🔊 Enable audio playback", value=False)

    st.sidebar.markdown("---")
    if st.sidebar.button("Clear conversation"):
        st.session_state["messages"] = []
        st.session_state["chat_history"] = []
        st.experimental_rerun()

    # Show chat messages
    for msg in st.session_state["messages"]:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(content)

    # Chat input
    user_input = st.chat_input(f"Ask a health question in {language_name}...")

    if user_input:
        # Display user message
        st.session_state["messages"].append({"role": "user", "content": user_input})
        chat_history.append(f"User: {user_input}")

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                prompt = build_prompt(user_input, lang_instruction, chat_history)
                response = llm.invoke(prompt)
                bot_reply = response.content.strip()

                st.markdown(bot_reply)

                # Save message + history
                st.session_state["messages"].append(
                    {"role": "assistant", "content": bot_reply}
                )
                chat_history.append(f"Bot: {bot_reply}")
                st.session_state["chat_history"] = chat_history

                # Save to "DB"
                save_chat_turn(user_input, bot_reply, language_name)

                # Optional TTS as inline audio
                if enable_tts:
                    audio_bytes = run_tts_and_prepare_audio(bot_reply)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")


if __name__ == "__main__":
    main()
