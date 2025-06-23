from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

chat_history = []

# improvement => append user_input to chat_history,
# when invoking model pass chat_history and append result from LLM to chat_history
# so it will now have all context around all chats with it
while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input == "exit":
        break

    result = model.invoke(chat_history)
    chat_history.append(result.content)
    print("AI: ", result.content)


print(chat_history)

