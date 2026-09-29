"""A local browser chat for the SmolLM2 learning project."""

from pathlib import Path
from threading import Lock

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "base" / "SmolLM2-135M-Instruct"
# Match the baseline instruction when starting a fresh conversation.
SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "Answer the question about the Ramayana in one or two sentences. "
        'If you do not know, say "I don\'t know."'
    ),
}


@st.cache_resource
def load_model():
    """Load once and share the model across reruns, with serialized inference."""
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, local_files_only=True, dtype=torch.float32,
    ).to("cpu").eval()
    return tokenizer, model, Lock()


st.set_page_config(page_title="SmolLM2 · Ramayana chat", page_icon="💬")
st.title("Chat with SmolLM2")
st.caption("Explore King Dasharatha's family with your local model, before fine-tuning.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.button("New chat"):
    st.session_state.messages = []

use_history = st.toggle(
    "Use conversation history",
    value=False,
    help="Turn on to include the questions and answers shown above in your next prompt.",
)
if use_history:
    st.caption("Follow-ups include earlier questions and answers. Chats are not saved to disk.")
else:
    st.caption("Each question is answered independently. Earlier answers stay visible but are not sent to the model.")

with st.spinner("Loading your local model…"):
    try:
        tokenizer, model, model_lock = load_model()
    except Exception as error:
        st.error(f"Could not load the local model: {error}")
        st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("truncated"):
            st.caption("Answer reached the 96-token limit.")

question = st.chat_input("Ask a question, such as: Who is Rama's father?")
if question and question.strip():
    question = question.strip()
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in (st.session_state.messages if use_history else [])
    ]
    messages = [SYSTEM_MESSAGE.copy()] + history + [
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_length = inputs["input_ids"].shape[1]
    if input_length + 96 > model.config.max_position_embeddings:
        st.warning("This conversation is too long. Start a new chat or shorten your question.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                generation = GenerationConfig(
                    max_new_tokens=96,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                                  else tokenizer.eos_token_id),
                )
                with model_lock, torch.inference_mode():
                    output = model.generate(**inputs, generation_config=generation)
                generated_ids = output[0, input_length:].tolist()
                answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
                truncated = bool(generated_ids and generated_ids[-1] != tokenizer.eos_token_id)
            except Exception as error:
                st.error(f"Could not generate an answer: {error}")
                st.stop()
        st.markdown(answer)
        if truncated:
            st.caption("Answer reached the 96-token limit.")

    st.session_state.messages.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "truncated": truncated},
    ])
