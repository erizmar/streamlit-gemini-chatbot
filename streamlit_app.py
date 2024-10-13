from main import ChatBot
import streamlit as st
import random
import string

bot = ChatBot()
thread_id = ""

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
            # if thread_id == "":
            #     thread_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=7))
            response = generate_response(input, "TFD")
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)