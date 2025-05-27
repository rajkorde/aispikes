import base64
import time

import nest_asyncio  # type: ignore
from dotenv import load_dotenv
from openai import OpenAI
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright

assert load_dotenv()

nest_asyncio.apply()

with sync_playwright() as p:
    browser = p.chromium.launch(
        headless=False,
        chromium_sandbox=True,
        env={},
        args=["--disable-extensions", "--disable-file-system"],
    )
    page = browser.new_page()
    page.set_viewport_size({"width": 1024, "height": 768})
    page.goto("https://google.com")
    page.wait_for_timeout(10000)
    page.goto("https://cnn.com")
    page.wait_for_timeout(10000)


async def run_playwright():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            chromium_sandbox=True,
            env={},
            args=["--disable-extensions", "--disable-file-system"],
        )
        page = await browser.new_page()
        await page.set_viewport_size({"width": 1024, "height": 768})
        await page.goto("https://google.com")
        await page.wait_for_timeout(2000)
        await page.goto("https://cnn.com")
        await page.wait_for_timeout(2000)
        await browser.close()


await run_playwright()


client = OpenAI()

response = client.responses.create(
    model="computer-use-preview",
    tools=[
        {
            "type": "computer_use_preview",
            "display_width": 1024,
            "display_height": 768,
            "environment": "browser",  # other possible values: "mac", "windows", "ubuntu"
        }
    ],
    input=[{"role": "user", "content": "Goto nytimes.com and read the latest news"}],
    reasoning={"summary": "concise"},
    truncation="auto",
)

print(response.output)


def handle_model_action(page, action):
    """
    Given a computer action (e.g., click, double_click, scroll, etc.),
    execute the corresponding operation on the Playwright page.
    """
    action_type = action.type

    try:
        match action_type:
            case "click":
                x, y = action.x, action.y
                button = action.button
                print(f"Action: click at ({x}, {y}) with button '{button}'")
                # Not handling things like middle click, etc.
                if button != "left" and button != "right":
                    button = "left"
                page.mouse.click(x, y, button=button)

            case "scroll":
                x, y = action.x, action.y
                scroll_x, scroll_y = action.scroll_x, action.scroll_y
                print(
                    f"Action: scroll at ({x}, {y}) with offsets (scroll_x={scroll_x}, scroll_y={scroll_y})"
                )
                page.mouse.move(x, y)
                page.evaluate(f"window.scrollBy({scroll_x}, {scroll_y})")

            case "keypress":
                keys = action.keys
                for k in keys:
                    print(f"Action: keypress '{k}'")
                    # A simple mapping for common keys; expand as needed.
                    if k.lower() == "enter":
                        page.keyboard.press("Enter")
                    elif k.lower() == "space":
                        page.keyboard.press(" ")
                    else:
                        page.keyboard.press(k)

            case "type":
                text = action.text
                print(f"Action: type text: {text}")
                page.keyboard.type(text)

            case "wait":
                print(f"Action: wait")
                time.sleep(2)

            case "screenshot":
                # Nothing to do as screenshot is taken at each turn
                print(f"Action: screenshot")

            # Handle other actions here

            case _:
                print(f"Unrecognized action: {action}")

    except Exception as e:
        print(f"Error handling action {action}: {e}")


def get_screenshot(page):
    """
    Take a full-page screenshot using Playwright and return the image bytes.
    """
    return page.screenshot()


def computer_use_loop(instance, response):
    """
    Run the loop that executes computer actions until no 'computer_call' is found.
    """
    while True:
        computer_calls = [
            item for item in response.output if item.type == "computer_call"
        ]
        if not computer_calls:
            print("No computer call found. Output from model:")
            for item in response.output:
                print(item)
            break  # Exit when no computer calls are issued.

        # We expect at most one computer call per response.
        computer_call = computer_calls[0]
        last_call_id = computer_call.call_id
        action = computer_call.action

        # Execute the action (function defined in step 3)
        handle_model_action(instance, action)
        time.sleep(1)  # Allow time for changes to take effect.

        # Take a screenshot after the action (function defined in step 4)
        screenshot_bytes = get_screenshot(instance)
        screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")

        # Send the screenshot back as a computer_call_output
        response = client.responses.create(
            model="computer-use-preview",
            previous_response_id=response.id,
            tools=[
                {
                    "type": "computer_use_preview",
                    "display_width": 1024,
                    "display_height": 768,
                    "environment": "browser",
                }
            ],
            input=[
                {
                    "call_id": last_call_id,
                    "type": "computer_call_output",
                    "output": {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{screenshot_base64}",
                    },
                }
            ],
            truncation="auto",
        )

    return response
