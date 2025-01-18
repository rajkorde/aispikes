from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph  # type: ignore
from langgraph.prebuilt import ToolNode, tools_condition  # type: ignore

from weekend_agent.tools import add, divide, multiply

assert load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [add, multiply, divide]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)


def asssistant(state: MessagesState):
    return {"Messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", asssistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)

builder.add_edge("tools", "assistant")

graph = builder.compile()

display(Image(graph.get_graph(xray=True).draw_mermaid_png()))

messages = [
    HumanMessage(
        content="Add 3 and 4. Multiply the output by 2. Divide the output by 5",
        role="user",
    )
]

messages = [
    HumanMessage(
        content="What is the capital of france?",
        role="user",
    )
]
messages = graph.invoke({"messages": messages})

for m in messages["messages"]:
    m.pretty_print()
