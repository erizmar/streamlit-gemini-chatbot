from main import ChatBot
import streamlit as st
import random, string

bot = ChatBot()

st.set_page_config(page_title="Stateful RAG Chat Bot")
with st.sidebar:
    st.title('Stateful RAG Chat Bot')

# Function for generating LLM response
def generate_response(input, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    result = bot.app.invoke(
        {"input": (input)},
        config=config,
    )
    return result["answer"]

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, I'm your assistant, ask me anything"}]

if "thread_id" not in st.session_state.keys():
    st.session_state.thread_id = ""

# Click to generate new chat session
trigger = st.button("New Chat")

if trigger:
    st.session_state.thread_id = "CISO_".join(random.choices(string.ascii_uppercase + string.digits, k=7))
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, I'm your assistant, ask me anything"}]

if st.session_state.thread_id != "":
    st.write(f"Session ID: {st.session_state.thread_id}")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if input := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": input})
        with st.chat_message("user"):
            st.write(input)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Getting your answer from mystery stuff.."):
                response = generate_response(input, st.session_state.thread_id)
                st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)