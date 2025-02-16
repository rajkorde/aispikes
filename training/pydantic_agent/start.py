import asyncio
import os
import tracemalloc
from dataclasses import dataclass

import logfire
import nest_asyncio  # type: ignore
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

tracemalloc.start()
assert load_dotenv()


@dataclass
class Store:
    def get_customer_name(self, customer_id: int) -> str:
        return "John Doe"

    def get_customer_balance(self, customer_id: int, include_pending: bool) -> float:
        if include_pending:
            return 200.0
        return 100.0


class SupportDependencies(BaseModel):
    customer_id: int = Field(description="ID of the customer")
    store: Store = Field(description="Store for customer data")


class SupportResult(BaseModel):
    support_advice: str = Field(description="Advice returned to the customer")
    block_card: bool = Field(description="Whether to block the card")
    risk_score: int = Field(description="Risk score of the query", ge=0, le=10)


support_agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt=(
        "You are a support agent in our bank, give the "
        "customer support and judge the risk level of their query."
    ),
    deps_type=SupportDependencies,
    result_type=SupportResult,
)


@support_agent.system_prompt
def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = ctx.deps.store.get_customer_name(ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


@support_agent.tool
def customer_balance(
    ctx: RunContext[SupportDependencies], include_details: bool
) -> float:
    return ctx.deps.store.get_customer_balance(ctx.deps.customer_id, include_details)


async def main():
    deps = SupportDependencies(
        customer_id=1,
        store=Store(),
    )

    result = await support_agent.run("What is my balance?", deps=deps)
    print(result.data)

    result = await support_agent.run("I just lost my card!", deps=deps)
    print(result.data)


if __name__ == "__main__":
    nest_asyncio.apply()
    logfire.configure(token=os.environ["LOGFIRE_TOKEN"])
    asyncio.run(main())
