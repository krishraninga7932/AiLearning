from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter
from langchain_text_splitters import RecursiveJsonSplitter
import json
import requests



# from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("../3.2-DataIngestion/attention.pdf")
pdf_docs=loader.load()
# print(pdf_docs)



# how to recursively split text by characters
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
# final_documents=text_splitter.split_documents(pdf_docs)
# print(final_documents[0])
# print(final_documents[1])




# from langchain_community.document_loaders import TextLoader
loader=TextLoader("../3.2-DataIngestion/speech.txt")
docs=loader.load()
# print(docs)



# speech=""
# with open("../3.2-DataIngestion/speech.txt") as f:
#     speech=f.read()
# print(speech)


# text_splitter=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
# text=text_splitter.create_documents([speech])
# print(text)



# from langchain_text_splitters import CharacterTextSplitter
# text_splitter=CharacterTextSplitter(separator="\n\n",chunk_size=100,chunk_overlap=20)
# text=text_splitter.split_documents(pdf_docs)
# print(text)



# HTML text splitter
# from langchain_text_splitters import HTMLHeaderTextSplitter

# html_string='''
#     <!DOCTYPE html>
# <html>

# <head>
#     <title>LangChain Tutorial</title>
# </head>

# <body>

#     <h1>Introduction</h1>

#     <p>
#         LangChain is a framework for building LLM applications.
#     </p>

#     <p>
#         It provides tools for document loading, splitting, embeddings,
#         vector databases, retrievers, memory and agents.
#     </p>

#     <h2>Document Loaders</h2>

#     <p>
#         Document loaders are used to read data from PDFs, text files,
#         HTML pages and many other sources.
#     </p>

#     <h2>Text Splitters</h2>

#     <p>
#         Text splitters divide long documents into smaller chunks so that
#         embeddings and retrieval become more efficient.
#     </p>

#     <h3>CharacterTextSplitter</h3>

#     <p>
#         CharacterTextSplitter splits text using a fixed separator.
#     </p>

#     <h3>RecursiveCharacterTextSplitter</h3>

#     <p>
#         RecursiveCharacterTextSplitter tries multiple separators and
#         preserves the document structure more effectively.
#     </p>

#     <h1>Embeddings</h1>

#     <p>
#         Embeddings convert text into numerical vectors that capture
#         semantic meaning.
#     </p>

# </body>

# </html>
# '''


# headers_to_split_on=[
#     ("h1","Header 1"),
#     ("h2","Header 2"),
#     ("h3","Header 3"),
#     # ("title","Title"), it will no print as HTMLHeaderTextSplitter is only used for spltting the header tag
# ]

# html_splitter=HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# html_header_splits=html_splitter.split_text(html_string)
# print(html_header_splits)


# HTML text splitter for url
# url = "https://plato.stanford.edu/entries/goedel/"

# headers_to_split_on=[
#     ("h1","Header 1"),
#     ("h2","Header 2"),
#     ("h3","Header 3"),
#     ("h4","Header 4"),
# ]

# html_splitter=HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
# html_header_splits=html_splitter.split_text_from_url(url)
# print(html_header_splits)




# JSON data splitter
# import json
# import requests
# from langchain_text_splitters import RecursiveJsonSplitter

# json_data=requests.get("https://api.smith.langchain.com/openapi.json").json()
# print(json_data)

# json_splitter=RecursiveJsonSplitter(max_chunk_size=300)
# json_chunks=json_splitter.split_json(json_data)
# print(json_chunks)

# for chunk in json_chunks[:3]:
#     print(chunk)


# The slitter can also output documents
# docs=json_splitter.create_documents(texts=[json_data])
# for doc in docs[:3]:
#     print(doc)


# texts=json_splitter.split_text(json_data)
# print(texts[0])
# print(texts[1])

