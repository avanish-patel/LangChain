from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# larger value of dimension will all it to capture more context
embedding = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=32
)

# embed_query to embed single statement
result = embedding.embed_query("Delhi is the capital of India")

print(str(result))