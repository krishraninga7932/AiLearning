import os 
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# google gemini
# models=init_chat_model("google_genai:gemini-3.5-flash")
# res=models.invoke("Who is Ganesh?")
# print(res.content)
# models=ChatGoogleGenerativeAI("google_genai:gemini-3.5-flash")

# models = ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash"
# )
# res=models.invoke("Who is Ganesh?")
# print(res.content)


# groq
models=init_chat_model("groq:openai/gpt-oss-120b")
# res=models.invoke("Who is Ganesh?")
# print(res.content)




# streaming
# for chunk in models.stream("Write a 1000 words para for AI"):
#     print(chunk.text,end="",flush=True)


# batch
res=models.batch(
    ["Why do parrots have colorfull feathers?",
    "How do airplanes fly",
    "what is quantum computing"
    ],
    config={
        'max_concurrency':5,
    }
)
print(res)