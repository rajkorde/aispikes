from typing import Optional

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Event(BaseModel):
    name: str = Field(description="Name of the Event")
    description: Optional[str] = Field(description="Brief description of the Event")
    date: str = Field(description="Date of the event")


class Events(BaseModel):
    events: list[Event] = Field(default_factory=list, description="List of events")


class EventExtracter:
    _extract_instructions = """
        You will be given a context that has text extracted from a website that contains a list of events and your job is to extract events from the list in a specific format. 

        Context: {context}
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        base_llm = ChatOpenAI(model=model_name)
        self.llm = base_llm.with_structured_output(Events)

    def extract_events(self, content: str) -> list[Event]:
        event_list = self.llm.invoke(
            [
                SystemMessage(
                    content=EventExtracter._extract_instructions.format(context=content)
                )
            ]
        )
        assert isinstance(event_list, Events)
        return event_list.events if event_list.events else []
