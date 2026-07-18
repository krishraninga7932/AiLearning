from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.prompts import MessagesPlaceholder
from dotenv import load_dotenv
load_dotenv()

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


db_path = "./chroma_db"
if os.path.exists(db_path):
    print("Loading existing Chroma database...")

    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
else:
    print("Creating new Chroma database...")

    loader=PyPDFLoader("./data/thinkpython.pdf")

    docs=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splits=text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=db_path
    )

retriever=vectorstore.as_retriever()
# results=retriever.invoke("What is python?")

groq_api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=groq_api_key,model="openai/gpt-oss-120b")


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following retrieved context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Keep the answer concise.\n\n"
    "{context}"
)

prompt=ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
])


question_answer_chain=create_stuff_documents_chain(llm,prompt)

contextualize_q_system_prompt=(
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt=ChatPromptTemplate.from_messages([
    ("system",contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever=create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)
rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

response=rag_chain.invoke(
    {
        "input":"Who build python and when?",
        "chat_history":[]
    }
)

# print(response)

chat_history=[]
question="Who built python and when?"

chat_history.append(
    HumanMessage(content=question)
)
chat_history.append(
    AIMessage(content=response["answer"])
)

# print(chat_history)

