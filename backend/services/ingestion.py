import re
import os
import tempfile
from fastapi import UploadFile
from langchain_community.document_loaders import TextLoader
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pytesseract
from pdf2image import convert_from_path

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                   model_kwargs={"local_files_only": True})

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
    temp_fd, temp_file_path = tempfile.mkstemp(suffix=file_extension)
    with os.fdopen(temp_fd, 'wb') as f:
        f.write(await file.read())
        
    try:
        # 1. Load documents using LangChain Wrappers
        if file_extension == ".pdf":
            loader = PyMuPDF4LLMLoader(temp_file_path, mode="page")
            docs = loader.load()
            
            for doc in docs:
                # If PyMuPDF couldn't extract meaningful text, assume it's a scanned image
                if len(doc.page_content.strip()) < 50:
                    page_num = doc.metadata.get("page", 0)
                    
                    # pdf2image uses 1-based indexing for page numbers
                    images = convert_from_path(
                        temp_file_path, 
                        first_page=page_num + 1, 
                        last_page=page_num + 1
                    )
                    
                    if images:
                        ocr_text = pytesseract.image_to_string(images[0])
                        # Replace the empty/short content with OCR'd text
                        doc.page_content = ocr_text + "\n"
                        # Tag it so you know which pages used OCR (great for debugging!)
                        doc.metadata["ocr_applied"] = True
                        
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
            separators=[
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
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
            try:
                os.remove(temp_file_path)
            except PermissionError:
                print(f"Warning: Windows locked {temp_file_path}. It will be cleaned up by the OS later.")