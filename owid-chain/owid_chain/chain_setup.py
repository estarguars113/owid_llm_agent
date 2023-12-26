# agent config
from langchain.agents import (
    initialize_agent,
    AgentExecutor,
    Tool
)
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

from langchain.tools import (
    BaseTool,
    StructuredTool
)

# memory management
from langchain.prompts.chat import MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

# chain pipelines
from langchain.chains import SequentialChain

from typing import Tuple, Dict

# community defined tools
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain.utilities import ArxivAPIWrapper

# custom defined tools
from tools_wrappers import OWIDAPIWrapper

class Config():
    """
    Contains the configuration of the LLM.
    """
    model = 'gpt-3.5-turbo-16k-0613'
    llm = ChatOpenAI(temperature=0, model=model)

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
    cfg = Config()
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

    prompt = """
            
            For the following input, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            There can only be two types of chart, "bar" and "line".

            If the data obtained from the tool includes a dataframe, include it in the answer:
            {"df: "answer"}
        
            Include all the associated description and metadata using the following format
            {"metadata": metadata}

            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}

            Return all output as a string.

            All strings in "columns" list and data list, should be in double quotes,

            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}


            Below is the query.
            Query:  {f}
        """
       
    FORMAT_INSTRUCTIONS = """Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    Thought: I now know the final answer based on my observation
    Final Answer: the final answer to the original input question is the Observation"""
    
    SUFFIX = """
    Question: {input}
    Thought:{agent_scratchpad}"""
    # agent_kwargs = agent_kwargs.update({
    #     'prefix': prompt,
    #     'format_instructions':FORMAT_INSTRUCTIONS,
    #     'suffix':SUFFIX,
    #     'input_variables': 'query'
    # })
    
    return initialize_agent(
        tools, 
        cfg.llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True, 
        agent_kwargs=agent_kwargs,
        memory=memory,
        prefix=prompt,
        #return_intermediate_steps=True
    )
        