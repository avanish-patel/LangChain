from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=0.5, max_completion_tokens=20)

result = model.invoke("What is a capital of India?")

print(result.content)

