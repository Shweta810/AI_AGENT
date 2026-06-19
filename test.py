# test.py

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

while True:
    query = input("Ask: ")

    response = llm.invoke(query)

    print("\n")
    print(response.content)
    print("\n")