import asyncio
from langchain_core.tools import tool
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI


@tool
def qna_icj_sa_case(user_query):
    """Use this to get relavent information on ICJ case on Isreal Genocide"""

    """
    loader = PyPDFLoader("docs/South-Africa-v-Israel.pdf")
    pages = loader.load_and_split()
    SA_ICJ_CASE_VDB = Chroma.from_documents(
        pages,
        HuggingFaceEmbeddings(),
        persist_directory="docs/icj_israel_genocide_case",
    )
    """
    SA_ICJ_CASE_VDB = Chroma(
        persist_directory="docs/icj_israel_genocide_case",
        embedding_function=HuggingFaceEmbeddings(),
    )
    return SA_ICJ_CASE_VDB.similarity_search(user_query, k=3)


llm = ChatOpenAI(model="gpt-4", temperature="0", streaming=True)
agent_tools = [qna_icj_sa_case]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


tool_node = ToolNode(agent_tools)

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a useful assistnat. Answer user query based on the tools you have access to.",
        ),
        ("user", "{input}"),
    ]
)


assistant_runnable = primary_assistant_prompt | llm.bind_tools(agent_tools)


# Define the function that determines whether to continue or not
def should_continue(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
async def call_model(state: State, config):
    message_memory = list()
    messages = state["messages"]
    if len(messages) >= 6:
        message_memory = messages[-6:]
    else:
        message_memory = messages
    # print("\nPrinting message..\n")
    # print(message_memory)
    response = await assistant_runnable.ainvoke(message_memory, config=config)
    # We return a list, because this will get added to the existing list
    return {"messages": response}


def create_state_graph():
    """
    Creates and returns a state graph with predefined nodes, edges, and memory checkpointing.

    Returns:
        StateGraph: A compiled state graph with nodes and edges set up.
    """

    builder = StateGraph(State)

    # Define nodes: these do the work
    builder.add_node("assistant", call_model)
    builder.add_node("action", tool_node)

    # Define edges: these determine how the control flow moves
    builder.set_entry_point("assistant")
    builder.add_conditional_edges(
        "assistant",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )
    builder.add_edge("action", "assistant")

    # The checkpointer lets the graph persist its state
    memory = AsyncSqliteSaver.from_conn_string("docs/memory/checkpoints.sqlite")
    graph = builder.compile(checkpointer=memory)
    return graph


lg_agent = create_state_graph()
