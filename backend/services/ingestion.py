import base64
import io
import pickle
import re
import os
import tempfile
import traceback
from fastapi import UploadFile
import fitz
from pdf2image import convert_from_path
from fastapi import UploadFile, BackgroundTasks
import asyncio

from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab", quiet=True)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5",
                                   model_kwargs={"local_files_only": False})

vector_store = Chroma(
    persist_directory="./data/chroma", 
    embedding_function=embeddings
)

vision_llm = ChatOllama(model="glm-ocr", temperature=0, 
                        #base_url="http://rag-ollama:11434"
                        )

def custom_word_tokenizer(text: str) -> list[str]:
    """Word-level tokenization using NLTK to enhance BM25Plus retrieval."""
    return word_tokenize(text.lower())

async def perform_ollama_ocr_async(img_b64: str) -> str:
    """Passes the base64 image to the local Ollama vision model Asynchronously."""
    message = HumanMessage(
        content=[
            {
                "type": "text", 
                "text": "Text Recognition: Extract all text, tables, and mathematical formulas from this image. Format everything in clean Markdown. Wrap inline math in $ and block math in $$. Do not include any other conversational text."
            },
            {
                "type": "image_url", 
                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
            }
        ]
    )
    # Use ainvoke instead of invoke for non-blocking execution
    response = await vision_llm.ainvoke([message])
    return response.content # type: ignore

def clean_text(text: str) -> str:
    """Basic text cleaning: remove null bytes, fix excessive whitespace and newlines."""
    text = text.replace('\x00', '')  # Remove null bytes sometimes found in PDFs
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def rebuild_bm25_index_background(tenant_id: str):
    """Runs silently in the background after the API has already responded to the user."""
    print(f"Background Task Started: Rebuilding BM25 for Tenant {tenant_id}...")
    
    bm25_dir = "./data/bm25"
    os.makedirs(bm25_dir, exist_ok=True)
    
    # 2. CREATE A HARD LOG FILE (To catch silent crashes)
    log_file = os.path.join(bm25_dir, f"bg_debug_{tenant_id}.log")
    
    try:
        results = vector_store.get(where={"tenant_id": tenant_id})
        
        all_tenant_docs = [
            Document(page_content=txt, metadata=meta) 
            for txt, meta in zip(results['documents'], results['metadatas'])
        ]
        
        if all_tenant_docs:
            with open(log_file, "a") as f:
                f.write("3. Building BM25 Index (If this is the last line, rank_bm25 is missing)\n")
            
            bm25_retriever = BM25Retriever.from_documents(
                all_tenant_docs,
                preprocess_func=custom_word_tokenizer
            )
            
            bm25_dir = "./data/bm25"
            os.makedirs(bm25_dir, exist_ok=True)
            
            # --- THE FIX: Atomic Writes ---
            temp_bm25_path = os.path.join(bm25_dir, f"temp_bm25_{tenant_id}.pkl")
            final_bm25_path = os.path.join(bm25_dir, f"bm25_{tenant_id}.pkl")
            
            # 1. Write to a temporary file first so we don't break active queries
            with open(temp_bm25_path, 'wb') as f:
                pickle.dump(bm25_retriever, f)
                
            # 2. Instantly swap the temp file to the real filename (Atomic Operation)
            os.replace(temp_bm25_path, final_bm25_path)
            
        with open(log_file, "a") as f:
                f.write("4. SUCCESS: BM25 pkl file saved and ready for queries!\n")
                
        print(f"Background Task Complete: BM25 Updated for Tenant {tenant_id}!", flush=True)
        
    except Exception as e:
        # CAPTURE THE EXACT ERROR TRACEBACK
        error_trace = traceback.format_exc()
        print(f"CRITICAL BG ERROR: {error_trace}", flush=True)
        with open(log_file, "a") as f:
            f.write(f"\nCRASHED WITH ERROR:\n{error_trace}\n")

async def process_and_store_document(file: UploadFile, tenant_id: str, background_tasks: BackgroundTasks) -> int:
    """
    Parses the uploaded file, cleans the text, chunks it, and saves it to ChromaDB.
    Triggers BM25 rebuild as a background task.
    """
    file_extension = os.path.splitext(file.filename)[1].lower() # type: ignore
    
    temp_fd, temp_file_path = tempfile.mkstemp(suffix=file_extension)
    with os.fdopen(temp_fd, 'wb') as f:
        f.write(await file.read())
        
    try:
        if file_extension == ".pdf":
            loader = PyMuPDF4LLMLoader(temp_file_path, mode="page")
            docs = loader.load()
            
            pdf_document = fitz.open(temp_file_path)
            
            # --- OPTIMIZATION 3: Collect OCR Tasks for Async Batching ---
            ocr_tasks_data = []
            for doc in docs:
                if len(doc.page_content.strip()) < 50:
                    page_num = doc.metadata.get("page", 0)
                    page = pdf_document.load_page(page_num)
                    pix = page.get_pixmap(dpi=150)  
                    img_bytes = pix.tobytes("png")
                    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                    ocr_tasks_data.append((doc, img_b64))
            
            pdf_document.close()

            # --- OPTIMIZATION 4: Run OCR Concurrently with a Semaphore ---
            if ocr_tasks_data:
                # Adjust this number based on your GPU/RAM. 
                # 4 means it will process 4 scanned pages simultaneously!
                semaphore = asyncio.Semaphore(4) 

                async def process_ocr_page(doc_obj, b64_str):
                    async with semaphore:
                        ocr_text = await perform_ollama_ocr_async(b64_str)
                        doc_obj.page_content = ocr_text + "\n"
                        doc_obj.metadata["ocr_applied"] = "ollama-glm-ocr"

                # Await all the concurrent batch tasks
                await asyncio.gather(*(process_ocr_page(d, b) for d, b in ocr_tasks_data))

        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path, encoding="utf-8")
            docs = loader.load()
        else:
            raise ValueError("Unsupported file type. Please upload a .pdf or .txt file.")
            
        for doc in docs:
            if 'clean_text' in globals():
                doc.page_content = clean_text(doc.page_content)
            doc.metadata["source"] = file.filename
            doc.metadata["tenant_id"] = tenant_id

        # --- Stage 3: Two-Stage Chunking Strategy ---
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False 
        )
        
        md_header_splits = []
        for doc in docs:
            splits = markdown_splitter.split_text(doc.page_content)
            for split in splits:
                split.metadata.update(doc.metadata)
            md_header_splits.extend(splits)

        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=400, separators=[
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
        chunks = recursive_splitter.split_documents(md_header_splits)
        
        if chunks:
            # OPTIMIZATION 5: Asynchronous Vector Store Insert
            vector_store.add_documents(chunks)  # type: ignore
            
            # --- Fire off the Background Task! ---
            # The API will not wait for this function to finish before responding to the user.
            background_tasks.add_task(rebuild_bm25_index_background, tenant_id)
            
        return len(chunks)
        
    finally:
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except PermissionError:
                pass