from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")

documents = loader.load()

print(len(documents))
print(documents[0].page_content)