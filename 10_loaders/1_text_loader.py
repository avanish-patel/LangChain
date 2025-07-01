from langchain_community.document_loaders import TextLoader

# define loader using TextLoader
loader = TextLoader("cricket.txt", encoding="utf-8")

# load document using loader
documents = loader.load()

print(documents)

print(len(documents))

print(type(documents))

print(documents[0])

print(type(documents[0]))

print(documents[0].page_content)