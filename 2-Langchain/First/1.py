import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="gemma:2b"
)
# response = llm.invoke("What is ai")

# print(response.content)


# chat prompt tempelate
from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are an expert Ai engineer. Provide answers based on the questions"),
        ("user","{input}")
    ]
)



chain=prompt|llm
response=chain.invoke({"input":"Can you tell me about langsmith?"})
# print(response)


# str output parser (it is used to give direct answer instead of whole object)
from langchain_core.output_parsers import StrOutputParser
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

response=chain.invoke({"input":"Can you tell me about langsmith?"})
print(response)