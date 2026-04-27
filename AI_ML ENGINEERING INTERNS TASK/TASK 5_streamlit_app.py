
# streamlit_app.py
# Run with: streamlit run streamlit_app.py
# Install: pip install streamlit transformers

import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="MindEase Chatbot", page_icon="💚")
st.title("💚 MindEase — Mental Wellness Support")
st.caption("A safe space to talk. I'm here to listen.")
st.warning("This is an AI assistant, not a substitute for professional mental health care.")

@st.cache_resource
def load_model():
    # Load your fine-tuned model
    return pipeline("text-generation", model="./healthbot_model_final")

generator = load_model()

# Keep conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hi, I'm MindEase. I'm here to listen — what's on your mind today?"}
    ]

# Display conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if user_input := st.chat_input("Share what's on your mind..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            prompt = f"<|user|> {user_input} <|bot|>"
            output = generator(
                prompt, max_new_tokens=100,
                temperature=0.7, top_p=0.9,
                repetition_penalty=1.3, do_sample=True
            )
            response = output[0]["generated_text"].split("<|bot|>")[1].strip()
            response = response.replace("<|endoftext|>", "").strip()
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
