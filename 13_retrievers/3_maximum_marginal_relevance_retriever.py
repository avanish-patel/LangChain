from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# source documents
documents = [
    Document(page_content="LangChain makes it easy to work with LLMs"),
    Document(page_content="LangChain helps developers build LLM applications easily"),
    Document(page_content="Chroma is vector database optimized for LLM-based applications"),
    Document(page_content="Embeddings convert text into multidimensional vectors"),
    Document(page_content="OpenAI and HuggingFace provides powerful embedding models"),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone and more"),
    Document(
        page_content="LangChain is an open-source framework designed to simplify the development of applications powered by large language models (LLMs)"),
]

# initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# create the FAISS vector store
vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model
)

# enable MMR strategy in the retriever
retriever = vector_store.as_retriever(
    # enables MMR
    search_type="mmr",
    # k: top results, lambda_mult: relevance/diversity balance (0 to 1 - lower number will reduce the redundancy)
    search_kwargs={"k": 3, "lambda_mult": 0.1}
)

results = retriever.invoke("What is langchain?")

# print retrieved document content
# In result you will see how it's not taking repeated LangChain statements and capture more context with divers results
for i, doc in enumerate(results):
    print(f"\n ---- Result {i + 1} ---")
    print(f"Content: \n {doc.page_content}")
