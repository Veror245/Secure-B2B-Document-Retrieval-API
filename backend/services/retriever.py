from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

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

# A simple, robust RAG prompt
class StructuredAnswer(BaseModel):
    answer_markdown: str = Field(description="The detailed answer formatted strictly in Markdown. Use headers, bullet points, and bold text where appropriate.")
    is_relevant: bool = Field(description="True if the context contained enough information to answer the question, False otherwise.")

# Initialize the parser
structured_llm = llm.with_structured_output(StructuredAnswer)

# A simple, robust RAG prompt
PROMPT_TEMPLATE = """
You are an expert AI assistant. Use the following proprietary context to answer the user's question.
If the answer is not contained within the context, politely state that you do not have enough information. Do not hallucinate.
If mathematical formulas appear broken or missing in the context, do your best to infer them, but wrap all mathematical notation in `$` for inline math and `$$` for block math so it renders cleanly in Markdown.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

async def query_documents(query: str, tenant_id: str):
    # 1. Retrieve relevant chunks (CRITICAL: Enforcing RBAC via Chroma metadata filtering)
    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 4, # Fetch the top 4 most relevant chunks
            "filter": {"tenant_id": tenant_id} # The user can ONLY search their own docs
        }
    )
    
    docs = retriever.invoke(query)
    
    if not docs:
        return {
            "answer_markdown": "### No Documents Found\nI couldn't find any relevant documents in your workspace to answer this.",
            "is_relevant": False,
            "sources": []
        }

    # 2. Format the retrieved documents into a single context string
    context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    # 3. Call Gemini
    chain = prompt | structured_llm
    response = chain.invoke({"context": context_text, "question": query})
    
    # 4. Format the sources for transparency
    sources = [
        {
            "file": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "preview": doc.page_content[:150] + "..."
        }
        for doc in docs
    ]
    
    return {
        "answer_markdown": response.answer_markdown, # type: ignore
        "is_relevant": response.is_relevant, # type: ignore
        "sources": sources
    }