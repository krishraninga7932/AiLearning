from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain_core.runnables import RunnableLambda
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=groq_api_key,model="openai/gpt-oss-120b")
os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")




documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"}
    ),
    Document(
        page_content="Cats are independent animals that enjoy sleeping for long hours.",
        metadata={"source": "mammal-pets-doc"}
    ),
    Document(
        page_content="Parrots can mimic human speech and are highly intelligent birds.",
        metadata={"source": "birds-doc"}
    ),
]





# vector store
vectorstore=Chroma.from_documents(documents,embedding=embeddings)
# print(vectorstore.similarity_search("Dog"))

# print(vectorstore.similarity_search_with_score("parrot"))

retriever=RunnableLambda(vectorstore.similarity_search).bind(k=1)
# print(retriever.batch(["cat","dog"]))


ret=vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":1}
)

# print(ret.batch(["cat","dog"]))


# RAG
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

message="""
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt=ChatPromptTemplate.from_messages([("human",message)])

rag_chain={"context":retriever,"question":RunnablePassthrough()}|prompt|llm

res=rag_chain.invoke("tell me  about dogs")
print(res.content)


