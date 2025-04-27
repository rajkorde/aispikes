import uuid

import typer
from dotenv import load_dotenv
from rich.prompt import IntPrompt, Prompt
from src.story import Story, StoryCondition, Student

assert load_dotenv()

__version__ = "0.0.1"

app = typer.Typer()


def version_callback(value: bool):
    if value:
        print(f"Recipe-Items  CLI Version: {__version__}")
        raise typer.Exit()


def print_header():
    typer.secho("\n-----------------------", fg=typer.colors.GREEN, bold=True)
    typer.secho("Welcome to Storyteller!", fg=typer.colors.GREEN, bold=True)
    typer.secho("-----------------------\n", fg=typer.colors.GREEN, bold=True)


# get student info
def ask_student_questions() -> dict[str, str | int]:
    answers: dict[str, str | int] = {}
    try:
        answers["name"] = Prompt.ask("What is the name of the student?")
        answers["age"] = IntPrompt.ask("What is their age?")
        answers["interests"] = Prompt.ask("What are their interests?")
        answers["situation"] = Prompt.ask(
            "What is student situation that needs correcting?"
        )
        answers["guidance"] = Prompt.ask(
            "Any guidance for the story (eg use forest setting, use bright colors etc)? Hit enter for none"
        )
    except Exception as e:
        typer.secho(f"Error during input: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    return answers


@app.command()
def get_student_info():
    print_header()
    responses = ask_student_questions()

    typer.secho("\nYour responses:", fg=typer.colors.CYAN, bold=True)
    for key, value in responses.items():
        typer.echo(f"{key.capitalize()}: {value}")

    assert isinstance(responses["name"], str)
    assert isinstance(responses["interests"], str)
    assert isinstance(responses["age"], int)

    scenario_id = str(uuid.uuid4())
    student = Student(
        name=responses["name"],
        interests=responses["interests"],
        age=responses["age"],
        scenario_id=scenario_id,
    )

    assert isinstance(responses["situation"], str)
    assert isinstance(responses["guidance"], str | None)
    guidance = None if not responses["guidance"] else responses["guidance"]
    story_condition = StoryCondition(
        scenario_id=scenario_id,
        student=student,
        situation=responses["situation"],
        guidance=guidance,
    )

    print(f"{story_condition=}")


# get story condition


# get story


if __name__ == "__main__":
    app()
