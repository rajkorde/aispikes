from typing import Annotated

from IPython.display import Image, display
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph  # type: ignore
from langgraph.graph.message import add_messages  # type: ignore
from langgraph.graph.state import CompiledStateGraph  # type: ignore
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list[str], add_messages]


def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


def show_graph(graph: CompiledStateGraph):
    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        print("Could not display graph")
        pass


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "context": user_input}]}):
        for value in event.values():
            print("Assistant: ", value["messages"][-1].content)


builder = StateGraph(State)


llm = ChatOpenAI(model="gpt-4o-mini")

builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye")
            break
        stream_graph_updates(user_input=user_input)
    except Exception:
        user_input = "Tell me an interesting fact"
        print(f"User: {user_input}")
        stream_graph_updates(user_input=user_input)
        break
