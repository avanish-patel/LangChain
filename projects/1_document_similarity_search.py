from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Abraham Lincoln was a pivotal figure in American history, serving as the 16th President of the United States and leading the nation through the Civil War.",
    "Marie Curie, a groundbreaking scientist, was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different scientific fields.",
    "Neil Armstrong made history as the first human to walk on the moon, uttering the iconic phrase, That's one small step for man, one giant leap for mankind.",
    "Jane Austen was a renowned English novelist whose works, like Pride and Prejudice and Sense and Sensibility, continue to be popular and beloved for their witty observations of social customs.",
    "Nelson Mandela dedicated his life to fighting against apartheid in South Africa, becoming a powerful symbol of peace and reconciliation worldwide."
]

query = "Tell me about Neil Armstrong and his achievement to the moon"


# returns [[]]
documents_embedding = embedding.embed_documents(documents)

# returns []
query_embedding = embedding.embed_query(query)

# returns [[]]
# Cosine similarity is a measure of similarity between two non-zero vectors in an inner product space.
# It is calculated as the cosine of the angle between them, resulting in a value between -1 and 1.
scores = cosine_similarity([query_embedding], documents_embedding)

# get first item of scores, wrap with enumerated list that will assign index to each item in scores
# sort list based on 2nd parameter which is score
# get first item in the sorted list which has the highest score with its index
index, score = sorted(list(enumerate(scores[0])), key=lambda x:x[1])[-1]

# print results
print(query)
print(documents[index])
print("Similarity score: ", score)


