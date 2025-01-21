import datetime
from typing import Optional

from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph  # type: ignore
from langgraph.prebuilt import ToolNode, tools_condition  # type: ignore
from pydantic import BaseModel, Field

from weekend_agent.tools import scrape_webpage

assert load_dotenv()

url = "https://www.events12.com/seattle/"

doc = scrape_webpage(url)

extract_instructions = """
You will be given a context that has text extracted from a website that contains a list of events and your job is to extract events from the list in a specific format. 

Context: {context}

"""


class Event(BaseModel):
    name: str = Field(description="Name of the Event")
    description: Optional[str] = Field(description="Brief description of the Event")
    date: str = Field(description="Date of the event")


class Events(BaseModel):
    events: list[Event] = Field(default_factory=list, description="List of events")


llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_structured_output = llm.with_structured_output(Events)

response = llm_with_structured_output.invoke(
    [SystemMessage(content=extract_instructions.format(context=doc.page_content))]
)

print(response.events)

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [scrape_webpage]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with coming up with interesting things to do in Seattle"
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
