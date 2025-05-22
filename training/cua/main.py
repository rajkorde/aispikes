import os
import time
from typing import List, Union

import keyboard
import pyautogui
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langgraph import Graph
from langgraph.nodes import ActionNode, LLMNode
from openai.error import OpenAIError

assert load_dotenv()


class DesktopToolset:
    def click(self, x: int, y: int) -> str:
        try:
            pyautogui.moveTo(x, y)
            pyautogui.click()
            return f"Clicked at ({x}, {y})"
        except Exception as e:
            return f"Failed to click: {e}"

    def type_text(self, text: str) -> str:
        try:
            keyboard.write(text)
            return f"Typed text: {text}"
        except Exception as e:
            return f"Failed to type text: {e}"

    def press_key(self, key: str) -> str:
        try:
            keyboard.press_and_release(key)
            return f"Pressed key: {key}"
        except Exception as e:
            return f"Failed to press key: {e}"

    def read_file(self, path: str) -> str:
        try:
            with open(path, "r") as f:
                return f.read()[:1000]  # truncate for safety
        except Exception as e:
            return f"Failed to read file: {e}"


class PlanAndActDesktopAgent:
    def __init__(self, model_name: str = "gpt-4") -> None:
        os.environ.setdefault("OPENAI_API_KEY", "YOUR_API_KEY")
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.tools = DesktopToolset()

        self.planner = LLMNode[List[str]](
            name="planner",
            llm=self.llm,
            prompt_template=(
                "You are a desktop assistant.\n"
                "Break the task into numbered UI steps.\n"
                "Example: click(x, y), type('hello'), read_file('/path')\n"
                "Task: {task}"
            ),
        )

        self.actor = ActionNode[str, str](name="actor", action_fn=self._dispatch_action)

        self.graph = Graph(start_node=self.planner)
        self.graph.add_node(self.actor)
        self.graph.add_edge(self.planner, self.actor)

    def _dispatch_action(self, command: str) -> str:
        try:
            if command.startswith("click("):
                x, y = eval(command[6:-1])
                return self.tools.click(x, y)
            elif command.startswith("type("):
                text = eval(command[5:-1])
                return self.tools.type_text(text)
            elif command.startswith("press_key("):
                key = eval(command[10:-1])
                return self.tools.press_key(key)
            elif command.startswith("read_file("):
                path = eval(command[10:-1])
                return self.tools.read_file(path)
            else:
                return f"Unknown command: {command}"
        except Exception as e:
            return f"Execution error: {e}"

    def run(self, task: str) -> None:
        try:
            plan = self.planner.run(task)
        except OpenAIError as err:
            raise RuntimeError("Planning failed") from err

        for step in plan:
            print(f">> {step}")
            result = self.actor.run(step)
            print(result)
            time.sleep(0.5)  # throttle to avoid spamming inputs


if __name__ == "__main__":
    agent = PlanAndActDesktopAgent()
    agent.run("Open Notepad and type 'Hello from your agent!' then save the file.")
