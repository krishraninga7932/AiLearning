from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_community.document_loaders import WikipediaLoader
import bs4




# from langchain_community.document_loaders import TextLoader
# loader=TextLoader("speech.txt")
# text_docs=loader.load()
# print(text_docs)



# from langchain_community.document_loaders import PyPDFLoader
# loader=PyPDFLoader('attention.pdf')
# pdf_docs=loader.load()
# print(pdf_docs)



# from langchain_community.document_loaders import WebBaseLoader
# import bs4
# loader=WebBaseLoader(web_paths=("https://quotes.toscrape.com/",),
#                         bs_kwargs=dict(parse_only=bs4.SoupStrainer(
#                             class_=("quote")
#                         ))
#                      )
# web_docs=loader.load()
# print(web_docs)



# from langchain_community.document_loaders import ArxivLoader
# docs=ArxivLoader(query="Attention Is All You Need",load_max_docs=2).load()
# print(docs)



from langchain_community.document_loaders import WikipediaLoader
docs=WikipediaLoader(query="Generative AI",load_max_docs=2).load()
print(docs)