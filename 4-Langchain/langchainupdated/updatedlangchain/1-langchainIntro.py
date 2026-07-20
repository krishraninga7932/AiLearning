import os
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY")
)


def get_weather(city:str)->str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."


agent=create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are helpfull assistant.",
)
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in Mumbai?"
            }
        ]
    }
)

print(response)
# print(response["messages"][-1].content)