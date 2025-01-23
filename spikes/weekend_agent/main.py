from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph  # type: ignore
from langgraph.prebuilt import tools_condition  # type: ignore

from weekend_agent.date_extracter import DateExtracter
from weekend_agent.events import EventExtracter
from weekend_agent.scraper import scrape_webpage

assert load_dotenv()

url = "https://www.events12.com/seattle/"

doc = scrape_webpage(url)
assert doc and len(doc) > 0

events = EventExtracter().extract_events(doc)
assert events

date_extracter = DateExtracter()
date_extracter.get_date_range(events[10].date)

for event in events:
    date_range = date_extracter.get_date_range(event.date)
    print(f"Name: {event.name}")
    if event.description:
        print(f"Name: {event.description}")
    if date_range.start_date == date_range.end_date:
        print(f"Date: {date_range.start_date.strftime('%b %d, %Y')}")
    else:
        print(f"Date: {date_range.start_date.strftime('%b %d, %Y')}")
        print(f"Date: {date_range.end_date.strftime('%b %d, %Y')}")
    print("-----")


llm = ChatOpenAI(model="gpt-4o-mini")


sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with coming up with interesting things to do in Seattle"
)


def asssistant(state: MessagesState):
    return {"Messages": [llm.invoke([sys_msg] + state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", asssistant)
# builder.add_node("tools", ToolNode())

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
