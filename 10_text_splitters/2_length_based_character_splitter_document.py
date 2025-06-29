from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

# create loader
loader = PyPDFLoader("dl-curriculum.pdf")

# load docs using load() method
docs = loader.load()

# create splitter
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    separator="",
)

# result will be list of split document objects
result = splitter.split_documents(docs)

# print(result)
print(result[0].page_content)
