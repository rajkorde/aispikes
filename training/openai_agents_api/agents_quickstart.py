import asyncio

import nest_asyncio
from agents import Agent, GuardrailFunctionOutput, InputGuardrail, Runner
from agents.extensions.visualization import draw_graph
from dotenv import load_dotenv
from pydantic import BaseModel

assert load_dotenv()
nest_asyncio.apply()

# Hello World
agent = Agent(name="Assistant", instructions="You are a helpful assistant.")
results = Runner.run_sync(agent, "Write a haiku about python programming")
print(results.final_output)

# Tutors
history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly",
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)


class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str


guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)


async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )


triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[InputGuardrail(guardrail_function=homework_guardrail)],
)


async def main():
    result = await Runner.run(triage_agent, "what was the last chinese dynasty?")
    print(result.final_output)
    return result


result = asyncio.run(main())
draw_graph(triage_agent)