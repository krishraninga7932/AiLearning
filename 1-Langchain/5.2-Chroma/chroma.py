# building sample vector db
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


loader=TextLoader("speech.txt")
data=loader.load()
# print(data)

# split
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=0)
splits=text_splitter.split_documents(data)


embedding=OllamaEmbeddings(model="nomic-embed-text")
vectordb=Chroma.from_documents(documents=splits,embedding=embedding)
# print(vectordb)


query="What is Artificial Intelligence?"
res=vectordb.similarity_search(query)
# print(res[0].page_content)


# save to the disk
vectordb=Chroma.from_documents(documents=splits,embedding=embedding,persist_directory="./chroma_db")


# load the disk
db2=Chroma(persist_directory="./chroma_db",embedding_function=embedding)
docs=db2.similarity_search(query)
# print(docs[0].page_content)


# similarity search with score
sdocs=vectordb.similarity_search_with_score(query)
# print(sdocs)



# retriever option
retriever=vectordb.as_retriever()
print(retriever.invoke(query)[0].page_content)