import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()


groq_api_key=os.getenv("GROQ_API_KEY")
model=ChatGroq(model="openai/gpt-oss-120b",groq_api_key=groq_api_key)
# print(model)



messages=[
    SystemMessage(content="Translate the following from English to Sanskrit"),
    HumanMessage(content="Hello how are you")
]
# res=model.invoke(messages)
# print(res)

parser=StrOutputParser()
# answer=parser.invoke(res)
# print(answer)


# chain=model|parser
# answer=chain.invoke(messages)
# print(answer)


# prompt template
generic_template="Translate into the following {language}:"
prompt=ChatPromptTemplate.from_messages(
    [("system",generic_template),("user","{text}")]
)

result=prompt.invoke({"language":"Sanskrit","text":"jai ganesh"})

result.to_messages()
chain=prompt|model|parser
answer=chain.invoke({"language":"Sanskrit","text":"jai ganesh"})
print(answer)