import os
import httpx
import logging
import datetime
from pathlib import Path
import openai
from dotenv import load_dotenv
import asyncio
from typing import Tuple, Dict, Any

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=env_path)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("OPENAI_API_KEY is missing in environment variables.")
    exit(1)

OPEN_AI_BASE_URL = "https://api.deepseek.com"
openai_client = openai.OpenAI(api_key=api_key, base_url=OPEN_AI_BASE_URL)

WATER_DATA_URL = (
    "https://waterinfo.rws.nl/api/detail/get"
    "?locationSlug=Rotterdam(ROTT)"
    "&expertParameter=Waterhoogte___20Oppervlaktewater___20t.o.v.___20Normaal___20Amsterdams___20Peil___20in___20cm"
)
WATER_MESSAGES_URL = (
    "https://waterinfo.rws.nl/api/watermessage/getall"
    "?identifiers=86b36486-c6d7-4897-9416-f3d2852a1287"
    "&identifiers=a87e691e-aab9-4993-b7db-0b7f3b73dbcd"
    "&identifiers=154c8fc5-0e8a-44e0-83c1-6c81f9144742"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    "Referer": "https://waterinfo.rws.nl/",
}


async def get_json(url: str) -> Dict[str, Any]:
    """Fetches JSON data from a specified URL asynchronously.

    Args:
        url: The URL to fetch JSON data from

    Returns:
        Parsed JSON response as dictionary

    Raises:
        httpx.HTTPError: If the request fails or returns non-2xx status
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return response.json()


async def fetch_data() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fetches water level data and safety messages from Rijkswaterstaat API.

    Returns:
        Tuple containing water data dictionary and safety messages dictionary
    """
    logging.info("Fetching water and safety data...")
    water_data, water_messages = await asyncio.gather(
        get_json(WATER_DATA_URL),
        get_json(WATER_MESSAGES_URL),
    )
    return water_data, water_messages


def create_prompt(water_data: Dict[str, Any], water_messages: Dict[str, Any]) -> str:
    """Constructs LLM prompt with current water conditions and safety information.

    Args:
        water_data: Dictionary containing water level and related measurements
        water_messages: Dictionary containing official safety messages

    Returns:
        Formatted prompt string for LLM processing
    """
    water_level = water_data["latest"]["data"]
    wind = next((r for r in water_data["related"] if "Windsnelheid" in r["label"]), {})
    temp = next(
        (r for r in water_data["related"] if "Watertemperatuur" in r["label"]), {}
    )

    messages = "\n".join(
        f"- {msg['title']}: {msg['bannerText']}" for msg in water_messages["messages"]
    )

    return f"""
You are an assistant that gives advice about swimming in open water at Rijnhaven, Rotterdam.
You should consider water temperature, water level, wind speed, and official safety notices.

Here is today's data:
- Water temperature: {temp.get('data')} °C
- Water level: {water_level} cm (relative to NAP)
- Wind speed: {wind.get('data')} m/s
- Official safety messages:
{messages}

Please provide your recommendation in plain HTML format without any code block indicators or additional markdown. Include relevant emojis to enhance the message's friendliness and clarity. Make sure important information is emphasized with appropriate HTML tags (e.g., <strong> for bold text). Don't add the header, just the content.
Also, mention that this is AI-generated content based on Rijkswaterstaat data, and its accuracy should be double-checked. Include a link to <a href="https://waterinfo.rws.nl" target="_blank">Rijkswaterstaat</a> for more information.
""".strip()


def get_llm_response(prompt: str) -> str:
    """Generates swimming advice using LLM based on constructed prompt.

    Args:
        prompt: Formatted prompt containing current conditions and safety info

    Returns:
        LLM-generated response as HTML-formatted string
    """
    logging.info("Sending prompt to LLM...")
    response = openai_client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )
    return response.choices[0].message.content


def export_to_html(report: str, file_path: str = "index.html") -> None:
    """Generates HTML report file with styled content and timestamp.

    Args:
        report: HTML content string to include in the report
        file_path: Output path for the HTML file
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Rijnhaven Swimming Advice</title>
            <style>
                body {{
                    margin: 0;
                    padding: 2em;
                    font-family: "Segoe UI", sans-serif;
                    background: #f7fbfe;
                    color: #333;
                }}
                .container {{
                    max-width: 800px;
                    margin: auto;
                    background: #fff;
                    padding: 2em;
                    border-radius: 12px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
                }}
                h1 {{
                    color: #0077cc;
                    font-size: 2em;
                    margin-bottom: 0.5em;
                }}
                .timestamp {{
                    margin-top: 2em;
                    font-size: 0.9em;
                    color: #777;
                }}
                .contribute-notice {{
                    margin-top: 1.5em;
                    font-size: 0.9em;
                    color: #666;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏊 Rijnhaven Swimming Advice</h1>
                <div>{report.replace('\n', '<br>')}</div>
                <div class="contribute-notice">
                    Have ideas or want to contribute? Open issues on 
                    <a href="https://github.com/arianium/rws_data_ingester" target="_blank">GitHub</a> 
                    to help improve this service.
                </div>
                <div class="timestamp">Last updated: {now}</div>
            </div>
        </body>
        </html>
    """
    Path(file_path).write_text(html_content.strip(), encoding="utf-8")
    logging.info(f"HTML report saved to: {file_path}")


async def main() -> None:
    """Main workflow orchestrator fetching data, generating advice, and exporting report."""
    try:
        water_data, water_messages = await fetch_data()
        prompt = create_prompt(water_data, water_messages)
        report = get_llm_response(prompt)
        export_to_html(report)
    except Exception as e:
        logging.error(f"Something went wrong: {e}")


if __name__ == "__main__":
    asyncio.run(main())
