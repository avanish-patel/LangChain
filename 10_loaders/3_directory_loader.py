from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# this loads all pdf files in a single shot, which could be memory inefficient
documents = loader.load()

print(len(documents))
print(documents)