import os 
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()


os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

text="Test Document"
query_result=embeddings.embed_query(text)
print(query_result)

print("=======")

doc_res=embeddings.embed_documents([text,"is it test document"])
print(doc_res[1])
