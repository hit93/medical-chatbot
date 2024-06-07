from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#Extract data from the PDF
def load_doc(list_file_path, chunk_size, chunk_overlap):
    # Processing for one document only
    # loader = PyPDFLoader(file_path)
    # pages = loader.load()
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap = 50)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap)
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits



#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings()
    return embeddings