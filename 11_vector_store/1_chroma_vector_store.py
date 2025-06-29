# Chroma is a lightweight, open source vector database that is specially friendly for local development and small to medium scale production needs.
# DB -> Collection -> Docs
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Create Langchain documents for IPL Players

doc1 = Document(
    page_content="Virat Kohli is the highest run-scorer in the IPL's history, representing Royal Challengers Bangalore since the league's inception and leading them as captain. Despite his exceptional individual performances, he is yet to lift the IPL trophy as a player.",
    metadata={"team": "RCB"}
)

doc2 = Document(
    page_content="A highly successful captain in IPL history, Rohit Sharma has led Mumbai Indians to five IPL titles, showcasing his brilliant leadership skills. He's a consistent performer and a prolific opener who has been a pillar for Mumbai Indians for years.",
    metadata={"team": "MI"}
)

doc3 = Document(
    page_content="Widely regarded as one of the greatest IPL captains, MS Dhoni has led Chennai Super Kings (CSK) to multiple title victories and is celebrated for his strategic acumen and 'finisher' role. He has been with CSK since the league's beginning, except during their absence in 2016 and 2017 when he played for Pune.",
    metadata={"team": "CSK"}
)

doc4 = Document(
    page_content="Jasprit Bumrah is a key bowler for Mumbai Indians, known for his lethal yorkers and death-bowling prowess. He has been a consistent performer and a vital part of Mumbai Indians' success, being retained by the franchise multiple times.",
    metadata={"team": "CSK"}
)

doc5 = Document(
    page_content="A dynamic all-rounder, Jadeja has played for multiple IPL franchises, showcasing his skills with both bat and ball, and is a valuable asset to any team. His impactful performances have made him a crucial player in the league, known for his fielding prowess and ability to contribute significantly in crucial moments.",
    metadata={"team": "MI"}
)

# create main doc with all above documents
docs = [doc1, doc2, doc3, doc4, doc5]

# define vector store using Chroma
vector_store = Chroma(
    # using local model to embed docs
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    # giving name to directory where it's going to store data locally
    persist_directory="chroma_db",
    # name of collection inside DB that is going to hold our documents
    collection_name="ipl_players",
)

# add documents
vector_store.add_documents(docs)

# view documents
print(vector_store.get(include=["embeddings","documents", "metadatas"]))

# search documents
result = vector_store.similarity_search(
    query="Who among these are the bowler?",
    # k=2 means it will return 2 documents in result with matching query
    k=2
)

print(result)

# search with similarity score
similarity_score = vector_store.similarity_search_with_score(
    query="Who among these are the bowler?",
    k=2
)
# the lower score means better match
print(similarity_score)

# metadata filter
result_by_metadata = vector_store.similarity_search(
    query="",
    filter={"team": "MI"}
)

print(result_by_metadata)

# update document
update_doc1 = Document(
    page_content="Updated Virat Kohli :)",
    metadata={"team": "RCB"}
)

vector_store.update_document(
    document_id="2ffdbea5-e261-4b85-86cf-bd64c8611462",
    document=update_doc1
)

# view documents
print(vector_store.get(include=["embeddings","documents", "metadatas"]))

# delete document
vector_store.delete(ids=["2ffdbea5-e261-4b85-86cf-bd64c8611462"])


# view documents
print(vector_store.get(include=["embeddings","documents", "metadatas"]))

