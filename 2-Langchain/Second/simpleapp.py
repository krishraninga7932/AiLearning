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


# Data ingestion from the webstoe we need to scrape the data
from langchain_community.document_loaders import WebBaseLoader
loader=WebBaseLoader("https://docs.smith.langchain.com/tutorials/Administrators/manage_spend")
# print(loader)

docs=loader.load()
# print(docs[0].page_content[:1000])
# print(docs)


# Load data-->Docs-->Diveide our texts into chunks-->vectors-->vector embedding-->vector db
from langchain_text_splitters import RecursiveCharacterTextSplitter

textsplitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=textsplitter.split_documents(docs)



from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)



from langchain_community.vectorstores import FAISS

vectorstoredb=FAISS.from_documents(documents,embeddings)
# print(vectorstoredb)


query="Langsmith has two usage limits: total traces and extended"
result=vectorstoredb.similarity_search(query)
result[0].page_content

# retrieval chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

prompt=ChatPromptTemplate.from_template(
    
    """
    Answer the follwing question based only on the provided context: 
    <context>
    {context}
    </context>
    
    Question:
    {input}
    """ 
)

document_chain=create_stuff_documents_chain(llm,prompt)
# print(document_chain)



# from langchain_core.documents import Document
# document_chain.invoke({
#     "input":"Langsmith has two usage limits: total traces and extended",
#     "context":[Document(page_content="Langsmith has two usage limits: total traces and extended traces.These correspond to the two metrics we've been tracking on our usage graph.")]
# })


# Retriever
retriever=vectorstoredb.as_retriever( search_kwargs={"k": 1})
from langchain_classic.chains import create_retrieval_chain
retrieval_chain=create_retrieval_chain(retriever,document_chain)
# print(retrieval_chain)



# get the response from the LLM
response=retrieval_chain.invoke({"input":"What is LangSmith Observability?"})
print(response['answer'])