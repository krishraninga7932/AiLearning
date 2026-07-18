from langchain_community.embeddings import OllamaEmbeddings
embeddings=(
    OllamaEmbeddings(model="nomic-embed-text") #by default it uses llama 2
)

# print(embeddings)

vector=embeddings.embed_documents(
    [
        "Hello how are you?",
        "I am fine, what about you",
    ]
)

print("My len",len(vector[0]))