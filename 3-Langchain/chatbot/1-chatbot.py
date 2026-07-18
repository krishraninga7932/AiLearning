import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage,trim_messages
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
load_dotenv()


groq_api_key=os.getenv("GROQ_API_KEY")

model=ChatGroq(model="openai/gpt-oss-120b",groq_api_key=groq_api_key)

model.invoke([HumanMessage(content="Hi, My name is Krish and I am a Chief AI Engineer")])

model.invoke([
    HumanMessage(content="Hi, My name is Krish and I am a Chief AI Engineer"),
    AIMessage(content="Hello Krish! 👋\n\nGreat to meet you—Chief AI Engineer sounds like an exciting role. How can I assist you today? Whether you’re looking for technical advice, brainstorming ideas, discussing AI strategy, or just want to chat about the latest developments in the field, I’m here to help."),
    HumanMessage(content="Hey What's my name and what do i do?")
])


# message history
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.runnables import RunnableWithMessageHistory

store={}

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

with_message_history=RunnableWithMessageHistory(model,get_session_history)

config={"configurable":{"session_id":"chat1"}}

with_message_history.invoke(
    [HumanMessage(content="Hi, My name is Krish and I am a Chief AI Engineer")],
    config=config
)

# change the config--> session id
config1={"configurable":{"session_id":"chat2"}}
with_message_history.invoke(
    [HumanMessage(content="What's my name")],
    config=config1
)
with_message_history.invoke(
    [HumanMessage(content="My name is john")],
    config=config1
)
with_message_history.invoke(
    [HumanMessage(content="What is my name")],
    config=config1
)
# print(res.content)


# prompt template
# from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpfull assistant. Answer all the question to the best of your ability in {language}"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain=prompt|model
chain.invoke({"messages":[HumanMessage(content="Hi my name is Krish")],"language":"Hindi"})

with_message_history=RunnableWithMessageHistory(chain,get_session_history,input_messages_key="messages")

config={"configurable":{"session_id":"chat3"}}
with_message_history.invoke(
    {"messages":[HumanMessage(content="Hi my name is Krish")],"language":"Hindi"},
    config=config

)
with_message_history.invoke(
    {"messages":[HumanMessage(content="Whats my name")],"language":"Hindi"},
    config=config

)
# print(res.content)

trimmer=trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

trimmed=trimmer.invoke(messages)
# print(trimmed)

chain=(
    RunnablePassthrough.assign(messages=itemgetter("messages")|trimmer)|prompt|model
)
chain.invoke(
    {
    "messages":messages+[HumanMessage(content="What math problem do i asked for?")],"language":"English"
    }
)



# lets wrap this in message history
with_message_history=RunnableWithMessageHistory(chain,get_session_history,input_messages_key="messages")
config={"configurable":{"session_id":"chat4"}}

res=with_message_history.invoke(
    {"messages":messages+[HumanMessage(content="Whats my name")],"language":"English"},
    config=config
)
print(res.content)