import streamlit as st
from dotenv import load_dotenv

from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import ReActSingleInputOutputParser

from callbacks import StreamlitCallbackHandler
import json

from utilities import write_response

import os
load_dotenv()
from chain_setup import setup_agent

QUESTION_HISTORY: str = 'question_history'


def init_stream_lit():
    title = "Chat Functions Introduction"
    st.set_page_config(page_title=title, layout="wide")
    agent_executor: AgentExecutor = prepare_agent()
    st.header(title)
    if QUESTION_HISTORY not in st.session_state:
        st.session_state[QUESTION_HISTORY] = []

    intro_text()
    simple_chat_tab, historical_tab = st.tabs(["Simple Chat", "Session History"])
    with simple_chat_tab:
        user_question = st.text_input("Your question")
        with st.spinner('Please wait ...'):
            try:
                response= agent_executor.run(user_question, callbacks=[StreamlitCallbackHandler(st)])
                if response:
                    response = json.loads(response)
                    write_response(st, response)
            except Exception as e:
                st.error(f"Error occurred: {e}")
    with historical_tab:
        for q in st.session_state[QUESTION_HISTORY]:
            question = q[0]
            if len(question) > 0:
                st.write(f"Q: {question}")
                st.write(f"A: {q[1]}")


def intro_text():
    with st.expander("Click to see application info:"):
        st.write(
            f"""Ask questions to ber resolved with:
            - [Wikipedia](https://www.wikipedia.org/) Content
            - [Our world in Data](https://ourworldindata.org/)(Custom Tool)
            """
    )
        
@st.cache_resource()
def prepare_agent() -> AgentExecutor:
    return setup_agent()


if __name__ == "__main__":
    init_stream_lit()