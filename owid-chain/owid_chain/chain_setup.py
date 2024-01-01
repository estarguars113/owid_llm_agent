# agent config
from langchain.agents import (
    initialize_agent,
    AgentExecutor,
    AgentType,
    AgentOutputParser,
)
from langchain.chat_models import ChatOpenAI

from langchain.tools import (
    BaseTool,
    StructuredTool
)

from langchain.prompts import PromptTemplate

# memory management
from langchain.prompts.chat import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# chain pipelines
from langchain.chains import (
    LLMChain,
    SequentialChain
)

from typing import Tuple, Dict

# community defined tools
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain.utilities import ArxivAPIWrapper

# custom defined tools
from tools_wrappers import OWIDAPIWrapper

MODEL = 'gpt-3.5-turbo-16k-0613'
    

def setup_memory() -> Tuple[Dict, ConversationBufferMemory]:
    """
    Sets up memory for the open ai functions agent.
    :return a tuple with the agent keyword pairs and the conversation memory.
    """
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        
    }
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    return agent_kwargs, memory


def setup_agent() -> AgentExecutor:
    """
    Sets up the tools for a function based chain.
    We have here the following tools:
    - wikipedia
    - arxiv (scientiphical pappers)
    - our world in data
    """
    wikipedia = WikipediaAPIWrapper()
    arxiv = ArxivAPIWrapper()

    owid_api_wrapper = OWIDAPIWrapper()
    owid_api_wrapper.doc_content_chars_max=5000

    tools = [
        StructuredTool.from_function(
            func=owid_api_wrapper.run,
            name="Our-world-in-data",
            description="""
                provides data and research on a wide range of global issues, including health, education, poverty, and the environment. 
                The type of data you can expect to obtain from Our World in Data is diverse and covers various aspects of human development and well-being. 
            """,
            return_direct=True
        )
    ]
    agent_kwargs, memory = setup_memory()

    # extend agent config
    agent_kwargs = {
        **agent_kwargs,
        **{
            "input_variables": [
                "input",
                "agent_scratchpad"
            ],
            "verbose": True
        }}

    prompt_base = """

            answer the users question as best as possible, using the available tools.

            Reply only using the metadata from the tool, if it contains data, save it locally as json

            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}

            Return all output as a string.


            Below is the query.
            Query:  {question}
        """
    prompt = PromptTemplate(
        template=prompt_base,
        input_variables=["question"]
    )
    llm = ChatOpenAI(temperature=0, model=MODEL)
    
    return initialize_agent(
        tools, 
        llm=llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, 
        memory=memory,
        prefix=prompt,
        stop=["\nObservation:"]
     )
        