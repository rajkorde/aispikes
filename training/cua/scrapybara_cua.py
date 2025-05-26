import os

from dotenv import load_dotenv
from scrapybara import Scrapybara
from scrapybara.openai import OpenAI
from scrapybara.tools import BashTool, ComputerTool, EditTool

assert load_dotenv()

client = Scrapybara(api_key=os.environ["SCRAPYBARA_API_KEY"])

instance = client.start_ubuntu()

response = client.act(
    tools=[ComputerTool(instance), BashTool(instance), EditTool(instance)],
    model=OpenAI(),
    system="You are a webscraping agent",
    prompt="Scrape the latest events happening around seattle this weekend",
)
