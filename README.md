# TravelBuddy India

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-1f6feb)](https://www.langchain.com/langgraph)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-5B2EFF)](https://www.trychroma.com/)

An agentic AI travel assistant focused on Indian destinations. The project combines retrieval-augmented generation (RAG), tool calling, memory, and a Streamlit interface to answer travel planning questions in a grounded way.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Example Prompts](#example-prompts)
- [Testing and Validation](#testing-and-validation)
- [Troubleshooting](#troubleshooting)

## Overview

TravelBuddy India is built as a capstone project to demonstrate:

- LangGraph workflow orchestration with multiple nodes and conditional routing
- ChromaDB-backed retrieval over curated travel knowledge
- Conversation memory using thread-based checkpoints
- A date/season tool for time-aware travel suggestions
- Self-reflection style answer checking via a faithfulness gate

## Features

- Multi-route assistant: retrieval, tool, and memory-only paths
- 12 curated India travel knowledge documents in the local knowledge base
- Seasonal travel recommendations based on current month
- Sliding conversation context window for token efficiency
- Faithfulness scoring with retry logic for low-grounding responses
- Streamlit chat UI with session isolation per thread

## Project Structure

```text
travelbuddy_india/
├── agent.py                 # LangGraph state, nodes, routing, evaluation, app build
├── knowledge_base.py        # 12 travel documents used for retrieval
├── capstone_streamlit.py    # Streamlit chat interface
├── day13_capstone.ipynb     # Notebook for capstone submission
├── pyproject.toml           # Project metadata and dependencies
├── requirements.txt         # Pip-compatible dependency list
└── README.md
```

## Architecture

```text
User Input
  -> memory_node      (append history, extract name, reset per-turn state)
  -> router_node      (classify route: retrieve | tool | memory_only)
  -> retrieval_node   (RAG context) OR
     tool_node        (date/season advice) OR
     skip_node        (memory-only path)
  -> answer_node      (compose grounded response)
  -> eval_node        (faithfulness scoring; retry if score < 0.7)
  -> save_node        (persist assistant response)
```

## Tech Stack

- Python 3.11+
- LangGraph
- Google Gemini (via langchain-google-genai)
- ChromaDB
- Sentence Transformers (all-MiniLM-L6-v2)
- Streamlit
- python-dotenv
- RAGAS (for evaluation workflow)

## Getting Started

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd travelbuddy_india
```

### 2. Create a virtual environment

Using uv:

```bash
uv venv --python 3.11
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell
```

Using standard venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Alternative with uv:

```bash
uv pip install -r requirements.txt
```

### 4. Configure environment variables

Create a .env file in the project root and add:

```env
GOOGLE_API_KEY=your_google_ai_studio_api_key
```

You can generate the key from Google AI Studio:
https://aistudio.google.com/

### 5. Run the app

```bash
streamlit run capstone_streamlit.py
```

Default local URL: http://localhost:8501

## Usage

- Open the Streamlit app in your browser.
- Ask travel-related questions about Indian destinations, budget planning, timing, and logistics.
- Start a new conversation from the sidebar to reset chat state and thread context.
- For direct script testing, run:

```bash
python agent.py
```

## Example Prompts

Retrieval route:

- What is the best time to visit Ladakh?
- Tell me about Goa beaches and budget.
- What permits are needed for Northeast India?

Tool route:

- What month is it now and where should I travel in India?
- What is the current travel season in India?

Memory route:

- My name is Asha and I like trekking.
- Which trek would you recommend for me?

## Testing and Validation

- Run the quick local tests embedded in agent.py:

```bash
python agent.py
```

- Validate Streamlit startup:

```bash
streamlit run capstone_streamlit.py
```

- Check these behaviors manually:
- Route selection works for retrieval, tool, and memory-only prompts
- Faithfulness score appears and low-confidence answers trigger retry flow
- Session memory remains consistent within a thread

## Troubleshooting

- Missing API key error:
  - Ensure .env exists in project root and contains GOOGLE_API_KEY.
- First run feels slow:
  - Model and embedding initialization can take time on initial startup.
- Port already in use:
  - Run Streamlit on another port, for example:

```bash
streamlit run capstone_streamlit.py --server.port 8502
```

