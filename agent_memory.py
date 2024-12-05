from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage
# from langchain_cohere.chat_models import ChatCohere
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

# os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    # model="llama3-70b-8192",
    model="gemma2-9b-it",
    temperature=0.1,
    # max_tokens=2000,
    # top_p=0.9,
    # verbose=True,
)
# llm = ChatCohere(model="command-r-plus", temperature=0.3)
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add,divide,multiply]

llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.perform the calculation only once")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
# react_graph = builder.compile()

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
react_graph_memory = builder.compile(checkpointer=memory)

# Specify a thread
config = {"configurable": {"thread_id": "1"}}

from IPython.display import Image
graph_png = react_graph_memory.get_graph().draw_mermaid_png()

# Save to a file
with open("graph.png", "wb") as f:
    f.write(graph_png)

print("Graph saved as graph.png. Open the file to view.")

# Specify an input
messages = [HumanMessage(content="add 1 and 2")]
# Run
messages = react_graph_memory.invoke({"messages": messages},config)
for m in messages['messages']:
    m.pretty_print()