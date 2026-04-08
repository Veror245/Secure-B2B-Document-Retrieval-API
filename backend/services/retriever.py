import logging
import os
import pickle

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Set up logging so you can actually SEE the rewritten queries in your terminal!
logging.basicConfig()
logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)


# Re-initialize the same embeddings and vector store connection
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"local_files_only": True}
)

vector_store = Chroma(
    persist_directory="./data/chroma",
    embedding_function=embeddings
)

# Initialize the LLM (Requires GOOGLE_API_KEY environment variable)
# llm = ChatGoogleGenerativeAI(model="gemma-3-12b-it", temperature=0.2)

llm = ChatOllama(model="gemma4:31b-cloud", temperature=0.2)
mqllm = ChatGoogleGenerativeAI(model="gemma-3-4b-it", temperature=0.6) 

cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",  model_kwargs={"local_files_only": True})
hf_compressor = CrossEncoderReranker(model=cross_encoder, top_n=5) 


# A simple, robust RAG prompt
class StructuredAnswer(BaseModel):
    answer_markdown: str = Field(description="The detailed answer formatted strictly in Markdown. Use headers, bullet points, and bold text where appropriate.")
    is_relevant: bool = Field(description="True if the context contained enough information to answer the question, False otherwise.")

# Initialize the parser
# structured_llm = llm.with_structured_output(StructuredAnswer, method="json_schema")
parser = PydanticOutputParser(pydantic_object=StructuredAnswer)

# A simple, robust RAG prompt
PROMPT_TEMPLATE = """
You are an expert AI assistant operating in a **strict retrieval-augmented generation (RAG) environment**.

Your task is to answer the user's question using ONLY the provided context.

----------------------------------------
🧠 CORE RULES
----------------------------------------

1. **Context is the single source of truth**
   - Do NOT use prior knowledge
   - Do NOT guess or hallucinate
   - If the answer is missing → explicitly say so

3. **Mathematical Handling**
   - If formulas are incomplete or slightly broken, intelligently reconstruct them
   - Use `$...$` for inline math
   - Use `$$...$$` for block equations


----------------------------------------
🔍 ANALYSIS PROCESS (INTERNAL)
----------------------------------------

- Identify relevant sections of the context
- Extract key facts
- Synthesize into a coherent answer
- Do NOT include this reasoning in the final output

----------------------------------------
🧾 OUTPUT FORMAT (STRICT)
----------------------------------------

- You MUST follow the schema EXACTLY
- Output ONLY valid JSON
- Do NOT include any text outside the JSON
- Do NOT wrap JSON in markdown/code blocks

----------------------------------------
📦 FORMAT INSTRUCTIONS
----------------------------------------
{format_instructions}

----------------------------------------
📚 CONTEXT
----------------------------------------
{context}

----------------------------------------
❓ QUESTION
----------------------------------------
{question}
"""
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)



async def query_documents(query: str, tenant_id: str):
    
    # STEP 1: Dense Retrieval (Vector Search)
    chroma_retriever = vector_store.as_retriever(
        search_kwargs={"k": 5, "filter": {"tenant_id": tenant_id}}
    )
    
    # STEP 2: Sparse Retrieval (BM25 Keyword Search)
    bm25_path = f"./data/bm25_{tenant_id}.pkl"
    if os.path.exists(bm25_path):
        with open(bm25_path, 'rb') as f:
            bm25_retriever = pickle.load(f)
        
        # Ensure BM25 returns the same amount of chunks as Chroma
        bm25_retriever.k = 5
        
        # STEP 3: LangChain's Ensemble Retriever 
        # (This automatically handles Reciprocal Rank Fusion & Deduplication)
        base_retriever = EnsembleRetriever(
            retrievers=[chroma_retriever, bm25_retriever],
            weights=[0.5, 0.5] # Weigh keyword and semantic search equally
        )
    else:
        # Fallback if no BM25 index exists for this tenant yet
        base_retriever = chroma_retriever
    
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""Generate 5 diverse search queries for the given question.
Each query must target a different perspective:
- definition
- cause
- example
- application
- comparison

Provide these alternative questions separated by newlines. Do not number them or include any conversational text.

Question: {question}"""
    )
    
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=mqllm,
        prompt=QUERY_PROMPT
    )

    # STEP 4: LangChain's Contextual Compression Retriever
    # We pipe the Ensembled results into our direct HuggingFace Model!
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=hf_compressor,
        base_retriever=mq_retriever
    )
    
    # Invoke the full pipeline (Dense+Sparse -> RRF -> Cross-Encoder)
    docs = compression_retriever.invoke(query)
    
    if not docs:
        return {
            "answer_markdown": "### No Documents Found\nI couldn't find any relevant documents in your workspace to answer this.",
            "is_relevant": False,
            "sources": []
        }

    # Format the retrieved documents into a single context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    # Generate the structured response
    chain = prompt | llm | parser
    response = chain.invoke({"context": context_text, "question": query, "format_instructions": parser.get_format_instructions()})
    
    # Format the sources for transparency
    sources = [
        {
            "file": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "preview": doc.page_content[:150] + "..."
        }
        for doc in docs
    ]
    
    
    
    return {
        "answer_markdown": response.answer_markdown,
        "is_relevant": response.is_relevant,
        "sources": sources
    }