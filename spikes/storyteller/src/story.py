from pydantic import BaseModel, Field


class Student(BaseModel):
    scenario_id: str
    name: str
    interests: str
    age: int


class StoryCondition(BaseModel):
    scenario_id: str
    student: Student
    situation: str
    guidance: str | None = None


class Story(BaseModel):
    scenario_id: str
    student: Student
    condition: StoryCondition
    text: str = ""
    setting: str = ""
    characters: list[str] = Field(default_factory=list)
    key_events: list[str] = Field(default_factory=list)
    STORY_LIMIT: int = 250

    def create_story(self): ...

    def find_setting(self): ...

    def find_character(self): ...

    def find_events(self): ...
