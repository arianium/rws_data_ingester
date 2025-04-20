# 🌊 RWS Swimming Advice

This project fetches real-time open water data currently for Rijnhaven, Rotterdam, and generates a swimming safety report using a hosted LLM model. The report is saved as a static HTML file.

---

## What It Does

- Fetches water level, temperature, wind speed, and official notices from Rijkswaterstaat APIs.
- Generates a human-friendly swimming recommendation with a mainstream LLM.
- Exports the advice as an HTML page.

---

## Setup Guide

### 1. Requirements

- Python 3.8+

---

### 2. Clone + Install

```bash
git clone https://github.com/arianium/rijnhaven-swimming-advice.git
cd src/rijnhaven-swimming-advice
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 3. Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your-openai-key-here
```

### 4. Run the Script

To manually generate the report:

```bash
python src/rijnhaven_swimming_advice.py
```

Or use the helper script:

```bash
./generate.sh
```

---
## License

MIT — use it freely!

