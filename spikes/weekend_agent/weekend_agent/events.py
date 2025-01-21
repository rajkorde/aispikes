class EventExtracter:
    extract_instructions = """
        You will be given a context that has text extracted from a website that contains a list of events and your job is to extract events from the list in a specific format. 

        Context: {context}
    """
    
    def __init__(self, model_name: str):
        self.model_name = 
    
    def extract_events(content: str) -> Events:




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
