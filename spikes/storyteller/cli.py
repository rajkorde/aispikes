import typer
from dotenv import load_dotenv
from rich.prompt import Confirm, Prompt
from src.story import Story, StoryCondition, Student

assert load_dotenv()

__version__ = "0.0.1"

app = typer.Typer()


def version_callback(value: bool):
    if value:
        print(f"Recipe-Items  CLI Version: {__version__}")
        raise typer.Exit()


# get student info
def ask_student_questions() -> dict[str, str | bool]:
    answers: dict[str, str | bool] = {}
    try:
        answers["name"] = Prompt.ask("What is the name of the student?")
        answers["age"] = Prompt.ask("What is their age?")
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
    typer.secho("\n-----------------------", fg=typer.colors.GREEN, bold=True)
    typer.secho("Welcome to Storyteller!", fg=typer.colors.GREEN, bold=True)
    typer.secho("-----------------------\n", fg=typer.colors.GREEN, bold=True)
    responses = ask_student_questions()
    typer.secho("\nYour responses:", fg=typer.colors.CYAN, bold=True)
    for key, value in responses.items():
        typer.echo(f"{key.capitalize()}: {value}")


# get story condition


# get story


if __name__ == "__main__":
    app()
