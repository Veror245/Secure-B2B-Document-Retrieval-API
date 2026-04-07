import re
import os
import tempfile
from fastapi import UploadFile
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vector_store = Chroma(
    persist_directory="./data/chroma", 
    embedding_function=embeddings
)

def clean_text(text: str) -> str:
    """Basic text cleaning: remove null bytes, fix excessive whitespace and newlines."""
    text = text.replace('\x00', '')  # Remove null bytes sometimes found in PDFs
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

async def process_and_store_document(file: UploadFile, tenant_id: str) -> int:
    """
    Parses the uploaded file, cleans the text, chunks it, and saves it to ChromaDB.
    Returns the number of chunks created.
    """
    file_extension = os.path.splitext(file.filename)[1].lower() # type: ignore
    
    # LangChain loaders expect physical file paths, so we use a temp file for the UploadFile stream
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
        
    try:
        # 1. Load documents using LangChain Wrappers
        if file_extension == ".pdf":
            loader = PyMuPDFLoader(temp_file_path)
            docs = loader.load()
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)
            docs = loader.load()
        else:
            raise ValueError("Unsupported file type. Please upload a .pdf or .txt file.")
            
        # 2. Clean text and inject RBAC metadata
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata["source"] = file.filename  # Override temp file path with real name
            doc.metadata["tenant_id"] = tenant_id

        # 3. Chunking Strategy
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        # split_documents preserves the rich metadata extracted by PyMuPDFLoader
        chunks = splitter.split_documents(docs)
        
        # 4. Save to ChromaDB
        if chunks:
            vector_store.add_documents(chunks)
            
        return len(chunks)
        
    finally:
        # Always clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)