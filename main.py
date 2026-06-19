import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

# Import your custom tools
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

# 1. Initialize the Gemini Model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)


SYSTEM_PROMPT = """
You are an expert AI assistant similar to ChatGPT.

RULES:

1. Simple factual questions
Examples:
- Who is the Prime Minister of India?
- What is the capital of France?
Give short and direct answers (1-3 sentences).

2. Educational topics
Examples:  
- Explain Arrays
- Tell me about Machine Learning
- What is Cloud Computing?
Provide:
- Definition
- How it works
- Examples
- Advantages
- Disadvantages
- Applications
- Summary

3. Coding Questions
If user asks:
- Give code
- Write a program
- Solve this problem
- LeetCode questions
- DSA questions
ALWAYS provide:
1. Problem Explanation
2. Approach
3. Complete Working Code
4. Dry Run
5. Time Complexity
6. Space Complexity
Never skip the code. Always put code inside markdown code blocks.

4. Research Questions
For:
- Research
- Roadmaps
- Tutorials
- Projects
- AI
- MERN
- DSA
Provide detailed structured answers.

5. Interview Questions
Provide:
- Explanation
- Answer
- Important Notes

Always answer exactly what the user asks.
If code is requested, ALWAYS provide code.
"""


agent = create_agent(
    model=llm,
    tools=[search_tool, wiki_tool, save_tool],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver()  # Retains conversation memory
)


config = {"configurable": {"thread_id": "agent_session_1"}}

print("AI Agent Initialized successfully! Type 'exit', 'quit', or 'bye' to stop.\n")


while True:
    try:
        query = input("What can I help you with? ")

       
        if query.strip().lower() in ["exit", "quit", "bye"]:
            print("\nGoodbye!")
            break

        
        if not query.strip():
            continue

       
        response = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )

        print("\n" + "=" * 80)
        print("FINAL ANSWER")
        print("=" * 80 + "\n")

       
        final_msg = response["messages"][-1]
        content = final_msg.content

       
        if isinstance(content, str):
            final_answer = content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            final_answer = "\n".join(text_parts)
        else:
            final_answer = str(content)

      
        print(final_answer.strip())
        print("\n" + "-" * 80 + "\n")

    except KeyboardInterrupt:
      
        print("\n\nSession closed by user. Goodbye!")
        break
    except Exception as e:
        print(f"\nAn error occurred: {e}\n")
