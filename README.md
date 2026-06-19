# AI Agent Workspace

An advanced AI assistant built with LangChain v1 and Gemini-2.5-Flash that handles dynamic query routing based on structured system rules.

## Features
- Multi-turn conversation tracking using LangGraph's `InMemorySaver`.
- Clean terminal-rendering output interface.
- Tool integration for search and web queries.

## Setup Instructions

 Clone the repository:
   ```bash
   git clone [https://github.com/Shweta810/AI_AGENT.git](https://github.com/Shweta810/AI_AGENT.git)
   cd AI_AGENT


 1. Create and activate a virtual environment:
 python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

2.Install required packages:
pip install -r requirements.txt


3.Create a .env file and add your credentials:
GOOGLE_API_KEY=your_gemini_api_key_here

4.Run the agent:
python main.py




