from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

assert load_dotenv()


class QAPair(BaseModel):
    question: str
    answer: str


class Questions(BaseModel):
    questions: list[QAPair]

    def __str__(self) -> str:
        s = ""
        for qa in self.questions:
            s += f"Question: {qa.question}\nAnswer: {qa.answer}\n\n"
        return s


def deserialize(file_path: str, model: type[BaseModel]) -> BaseModel:
    try:
        json_str = Path(file_path).read_text()
        return model.model_validate_json(json_str)
    except Exception as e:
        raise ValueError(f"Failed to read or parse file: {e}")


def respond(
    message: str, chat_history: list[tuple[str, str]], questions: Questions
) -> list[tuple[str, str]]:
    """
    Generate a response using OpenAI API based on the user's message and conversation history.

    Args:
        message: The current message from the user
        chat_history: List of previous (user, assistant) message pairs

    Returns:
        Updated chat history with the new user message and assistant response
    """

    system_prompt = """
You are a helpful chat assistant. 

If the user question is similar to the list of questions provided below, respond with the answer provided.

If the user question is not similar to the list of questions provided below, respond normally as you would.

"""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    history_langchain = [SystemMessage(content=f"{system_prompt}\n{str(questions)}")]
    for msg in chat_history:
        if msg["role"] == "user":
            history_langchain.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_langchain.append(AIMessage(content=msg["content"]))

    history_langchain.append(HumanMessage(content=message))
    # Call the OpenAI API
    response = model.invoke(history_langchain)
    return response.content

    # Extract the assistant's response
    if (
        not response
        or not response.choices
        or response.choices[0].finish_reason != "stop"
    ):
        new_message = (message, "Error: Could not generate response")
    else:
        assistant_response = response.choices[0].message.content
        assert isinstance(assistant_response, str)
        new_message = (message, assistant_response)

    if chat_history:
        chat_history.append(new_message)
    else:
        chat_history = [new_message]
    return chat_history


questions = deserialize("data.json", Questions)


interface = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="Thoughtful AI Chatbot",
    description="Thoughtful AI Chatbot.",
)

interface.launch()
