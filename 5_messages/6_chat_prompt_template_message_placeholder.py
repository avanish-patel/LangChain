from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# crate chat template
chat_template = ChatPromptTemplate([
    ("system", "You are helpful customer support agent"),
    # insert chat history before new query
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

chat_history=[]
# load chat history (assume we are reading from DB)
with open("./5_messages/chat_history.txt") as f:
    chat_history.extend(f.readlines())

# print(chat_history)

# create prompt
prompt = chat_template.invoke({"chat_history":chat_history, "query": "Where is my refund?"})

print(prompt)
