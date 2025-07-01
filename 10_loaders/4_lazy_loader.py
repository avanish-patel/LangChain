from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="books",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

# will return generator of Documents
documents = loader.lazy_load()

for document in documents:
   print(document.metadata)