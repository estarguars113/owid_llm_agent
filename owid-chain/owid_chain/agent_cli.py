from prompt_toolkit import HTML, prompt, PromptSession
from prompt_toolkit.history import FileHistory

from langchain.input import get_colored_text
from dotenv import load_dotenv
from langchain.agents import AgentExecutor

import langchain
from callbacks import AgentCallbackHandler

import json
import os
load_dotenv()
from chain_setup import setup_agent

langchain.debug = True

if __name__ == "__main__":

    agent_executor: AgentExecutor = setup_agent()
    
    session = PromptSession(history=FileHistory(".agent-history-file"))
    while True:
        question = session.prompt(
            HTML("<b>Type <u>Your question</u></b>  ('q' to exit): ")
        )
        if question.lower() == 'q':
            break
        if len(question) == 0:
            continue
        try:
            response = agent_executor.run(question, callbacks=[AgentCallbackHandler()])
            response = json.loads(response)
            print(get_colored_text("Response: >>> ", "green"))
            print(get_colored_text(response, "green"))
        except Exception as e:
            print(get_colored_text(f"Failed to process {question}", "red"))
            print(get_colored_text(f"Error {e}", "red"))