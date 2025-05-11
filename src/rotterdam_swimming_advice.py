import json
import os
import httpx
import logging
import datetime
from pathlib import Path
import openai
from dotenv import load_dotenv
import asyncio
from typing import Mapping, Tuple, Dict, Any, Union
from bs4 import BeautifulSoup

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
RWS_HEADERS = {
    "Accept": "application/json",
    "Referer": "https://waterinfo.rws.nl/",
}

ZWM_WTR_HEADERS = {
    "Accept": "*/*",
    "Referer": "https://www.zwemwater.nl/",
}

BASE_SAFETY_URL = "https://www.zwemwater.nl/wp-content/themes/stuurlui/blocks/map/map-content.php?spotid="


ZWEMWATER_IDS = ["22003", "23762", "22005", "22001"]


def parse_zwemwater_html_to_dict(html_content: str) -> Dict[str, Any]:
    """Parses Zwemwater.nl-style HTML content into structured data."""
    try:
        soup = BeautifulSoup(html_content, "lxml")
        result = {}

        # Get location name
        place_heading = soup.find("h2")
        result["place"] = (
            place_heading.get_text(strip=True) if place_heading else "Unknown"
        )

        # Extract general info list (address, control, access, etc.)
        general_info = {}
        for li in soup.select("ul.spot-info li"):
            label = li.find("span")
            if label:
                key = label.get_text(strip=True).rstrip(":")
                text = (
                    li.get_text(strip=True)
                    .replace(label.get_text(strip=True), "")
                    .strip(": ")
                )
                general_info[key] = text
        result["general_info"] = general_info

        # Descriptive paragraphs
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        result["description"] = "\n".join(paragraphs)

        # Facilities
        features = [
            button.get_text(strip=True)
            for button in soup.select("ul.features button span.border-b")
        ]
        result["facilities"] = features

        # Optional: Chart headers
        chart_titles = [h4.get_text(strip=True) for h4 in soup.find_all("h4")]
        result["chart_titles"] = chart_titles

        return result
    except Exception as exc:
        return {"error": str(exc), "raw": html_content}


async def get_zwemwater_safety_data() -> Dict[str, Any]:
    """Fetches additional safety data from specified URLs asynchronously.

    Returns:
        Aggregated safety data dictionary
    """
    logging.info("Fetching additional safety data...")
    safety_data_tasks = [
        call_endpoint_and_get_content(f"{BASE_SAFETY_URL}{spotid}", ZWM_WTR_HEADERS)
        for spotid in ZWEMWATER_IDS
    ]
    safety_data_list = await asyncio.gather(*safety_data_tasks)

    safety_messages = []
    for html_data in safety_data_list:
        safety_messages.append(parse_zwemwater_html_to_dict(html_data))

    return {"safetyMessages": safety_messages}


async def call_endpoint_and_get_content(
    url: str, header: dict[str, str]
) -> Union[dict[str, Any], str]:
    """Fetches JSON data from a specified URL asynchronously.

    Args:
        url (str): The URL to fetch JSON data from.
        header (dict[str, str]): A dictionary of headers to include in the request.

    Returns:
        Union[dict[str, Any], str]: Parsed JSON response as a dictionary,
        or the raw response text if JSON decoding fails.

    Raises:
        httpx.HTTPError: If the request fails or returns a non-2xx status.
        json.JSONDecodeError: If the response cannot be parsed as JSON.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=header, timeout=10)
        response.raise_for_status()
        try:
            return response.json()
        except json.JSONDecodeError:
            return response.text


async def fetch_rws_data() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fetches water level data and safety messages from Rijkswaterstaat API.

    Returns:
        Tuple containing water data dictionary and safety messages dictionary
    """
    logging.info("Fetching water and safety data...")
    water_data, water_messages = await asyncio.gather(
        call_endpoint_and_get_content(WATER_DATA_URL, RWS_HEADERS),
        call_endpoint_and_get_content(WATER_MESSAGES_URL, RWS_HEADERS),
    )
    return water_data, water_messages


def create_prompt(
    rws_water_data: Mapping[str, Any],
    water_messages: Mapping[str, Any],
    zwemwater_safety_data: Mapping[str, Any],
) -> str:
    """
    Builds a prompt for the LLM, instructing it to generate HTML in two sections:
    1. Rijnhaven advice (based on Rijkswaterstaat data)
    2. Water safety notes (based on Zwemwater.nl data)
    """
    # Extract main measurements
    level = rws_water_data.get("latest", {}).get("data", "N/A")
    wind = next(
        (
            item
            for item in rws_water_data.get("related", [])
            if "Windsnelheid" in item.get("label", "")
        ),
        {},
    )
    temp = next(
        (
            item
            for item in rws_water_data.get("related", [])
            if "Watertemperatuur" in item.get("label", "")
        ),
        {},
    )

    # Format Rijkswaterstaat messages
    rws_msgs = water_messages.get("messages", [])
    rws_lines = [f"- {msg['title']}: {msg['bannerText']}" for msg in rws_msgs]
    rws_summary = "\n".join(rws_lines) if rws_lines else "- No official messages"

    prompt = f"""
    You are an assistant that provides swimming advice in Rotterdam.
    The language of the advice should be English except for names.

    Generate an HTML response with two clearly separated sections. 
    Keep vertical spacing between them minimal but clear. 
    Use CSS margin spacing, not multiple <br> tags, to separate sections.

    Do NOT include full HTML boilerplate like <!DOCTYPE html>, <html>, <head>, or <body>. 
    Only return the content inside a <div>.

    Important: Do NOT use triple backticks (```html) or any code block formatting in your output. 
    Just return raw HTML.
    1. Rijnhaven Advice:
    - Use the following data:
        - Water temperature: {temp.get('data', 'N/A')} °C
        - Water level: {level} cm (relative to NAP)
        - Wind speed: {wind.get('data', 'N/A')} m/s
        - Official safety messages from Rijkswaterstaat:
    {rws_summary}

    2. Water Safety Notes:
    - Based on recent data from Zwemwater.nl:
    {zwemwater_safety_data}

    Format the HTML in a clean and friendly manner. 
    Use <strong> for important facts, emojis to make it friendlier, and only single <br> for spacing. 
    Avoid <br><br> and excessive empty space.

    At the end, add a short note that this advice is AI-generated using public data sources 
    and include links to:
    - https://waterinfo.rws.nl
    - https://www.zwemwater.nl
    """.strip()

    return prompt


def get_llm_response(prompt: str) -> str:
    """Generates swimming advice using LLM based on constructed prompt.

    Args:
        prompt: Formatted prompt containing current conditions and safety info

    Returns:
        LLM-generated response as HTML-formatted string
    """
    logging.info("Sending prompt to LLM...")
    response = openai_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


def export_to_html(report: str, file_path: str = "index.html") -> None:
    """Generates HTML report file with styled content and timestamp.

    Args:
        report: HTML content string to include in the report
        file_path: Output path for the HTML file
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    html_content = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '    <meta charset="UTF-8">\n'
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "    <title>Rotterdam Swimming Advice</title>\n"
        "    <style>\n"
        "        body {\n"
        "            margin: 0;\n"
        "            padding: 2em;\n"
        '            font-family: "Segoe UI", sans-serif;\n'
        "            background: #f7fbfe;\n"
        "            color: #333;\n"
        "        }\n"
        "        .container {\n"
        "            max-width: 800px;\n"
        "            margin: auto;\n"
        "            background: #fff;\n"
        "            padding: 2em;\n"
        "            border-radius: 12px;\n"
        "            box-shadow: 0 4px 20px rgba(0,0,0,0.05);\n"
        "        }\n"
        "    </style>\n"
        "</head>\n"
        "<body>\n"
        f'    <div class="container">\n'
        f"        <h1>🏊 Rotterdam Swimming Advice</h1>\n"
        f'        <div>{report.replace(chr(10), "<br>")}</div>\n'
        '        <div class="contribution">\n'
        '            <p>If you have any suggestions or ideas, feel free to reach out to me on <a href="https://github.com/arianium/rws_data_ingester">GitHub</a>.</p>\n'
        "        </div>\n"
        f'        <div class="timestamp">Last updated: {now}</div>\n'
        "    </div>\n"
        "</body>\n"
        "</html>"
    )

    Path(file_path).write_text(html_content, encoding="utf-8")
    logging.info(f"HTML report saved to: {file_path}")


async def main() -> None:
    """Main workflow orchestrator fetching data, generating advice, and exporting report."""
    try:
        water_data, water_messages = await fetch_rws_data()
        zwemwater_safety_data = await get_zwemwater_safety_data()
        prompt = create_prompt(water_data, water_messages, zwemwater_safety_data)
        report = get_llm_response(prompt)
        export_to_html(report)
    except Exception as e:
        logging.error(f"Something went wrong: {e}")


if __name__ == "__main__":
    asyncio.run(main())
