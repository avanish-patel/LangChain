from langchain_community.retrievers import WikipediaRetriever

# Initialize the retriever (k results 2 means will return 2 docs in output)
retriever = WikipediaRetriever(top_k_results=2, lang="en")

# define query
query = "The cost of capital in business"

# invoke query
docs = retriever.invoke(query)

# print retrieved document content
for i, doc in enumerate(docs):
    print(f"\n ---- Result {i + 1} ---")
    print(f"Content: \n {doc.page_content}")
