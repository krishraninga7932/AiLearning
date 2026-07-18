from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

loader=TextLoader("speech.txt")

documents=loader.load()
text_Splitter=CharacterTextSplitter(chunk_size=200,chunk_overlap=30)
docs=text_Splitter.split_documents(documents)

# print(docs)

embeddings=OllamaEmbeddings(model="nomic-embed-text")
db=FAISS.from_documents(docs,embeddings)
# print(db)



# querying
query="What is Artificial Intelligence?"
res=db.similarity_search(query)
# print(res[0].page_content)


retriever=db.as_retriever()
# print(retriever.invoke(query))

docs_and_score=db.similarity_search_with_score(query)
# print(docs_and_score)



embedding_vector=embeddings.embed_query(query)
# print(embedding_vector)


docs_score=db.similarity_search_by_vector(embedding_vector)
# print(docs_score)


# saving and loading
db.save_local("faiss_index")

new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
docs=new_db.similarity_search(query)
# print(docs)