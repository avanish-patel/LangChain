from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily"),
    Document(page_content="Chroma is vector database optimized for LLM-based applications"),
    Document(page_content="Embeddings convert text into multidimensional vectors"),
    Document(page_content="OpenAI and HuggingFace provides powerful embedding models")
]

# initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# create vector store and add documents
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="my_collection"
)

# convert vector store into retriever, k:1 means return 1 result with most similarity
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# query retriever, return Document object
result = retriever.invoke("What is Chroma used for?")

# print retrieved document content
for i, doc in enumerate(result):
    print(f"\n ---- Result {i + 1} ---")
    print(f"Content: \n {doc.page_content}")
