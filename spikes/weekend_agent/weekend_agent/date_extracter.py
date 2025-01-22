from datetime import date
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class DateRange(BaseModel):
    start_date: date
    end_date: date

    class Config:
        @staticmethod
        def json_schema_extra(schema: dict[str, Any], model: type["DateRange"]) -> None:
            for field in schema.get("properties", {}).values():
                field.pop("format", None)


class DateExtracter:
    _extract_instructions = """
        You will be given some text that can contain either a single date or a date range. Your job is to extract the start and end date from this. The dates should strictly be in ISO 8601 format (YYYY-MM-DD).
        If there is a single date, then start and end date will be the same.
        If year is not provided, assume its the current year.

        Examples:

        Text: January 3, 2025
        start_date: 2025-01-03
        end_date: 2025-01-03

        Text: Jan 3, 2025
        start_date: 2025-01-03
        end_date: 2025-01-03

        Text: Jan. 3, 2025
        start_date: 2025-01-03
        end_date: 2025-01-03

        Text: Jan 5 - 12, 2025
        start_date: 2025-01-05
        end_date: 2025-01-12

        Text: January 28 - Feb. 5, 2025
        start_date: 2025-01-28
        end_date: 2025-02-05

        Text: {date_str}
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        base_llm = ChatOpenAI(model=model_name)
        self.llm = base_llm.with_structured_output(DateRange)

    def get_date_range(self, date_str: str) -> DateRange:
        date_range = self.llm.invoke(
            [
                SystemMessage(
                    content=DateExtracter._extract_instructions.format(
                        date_str=date_str
                    )
                )
            ]
        )
        assert isinstance(date_range, DateRange)
        return date_range
