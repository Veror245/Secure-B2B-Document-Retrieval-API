import os
import time
import requests
import pandas as pd
from datasets import load_dataset, Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
TEST_TENANT_ID = "biomed_tenant"

def run_end_to_end_test():
    print("1. Loading Medical Research Dataset (COVID-QA)...")
    # This dataset contains parsed PDF research papers (extremely noisy text, no tables)
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    
    # We will test 25 specific questions
    test_cases = dataset.select(range(25))
    
    print("\n2. Building the Medical Haystack...")
    # Because multiple questions can come from the same paper, we want to extract
    # UNIQUE papers to build a massive, noisy corpus.
    unique_contexts = set()
    
    # Get papers for our test questions
    for row in test_cases:
        # PubMed QA stores contexts as a list of strings inside a dictionary
        ctx = "\n".join(row['context']['contexts']) # type: ignore
        unique_contexts.add(ctx) # type: ignore
        
    # Inject 50 MORE random medical papers as pure distractors
    for row in dataset.select(range(25, 400)):
        ctx = "\n".join(row['context']['contexts']) # type: ignore
        unique_contexts.add(ctx)
        if len(unique_contexts) >= 50:
            break
            
    corpus_text = "# Biomedical Research Knowledge Base\n\n"
    for ctx in unique_contexts:
        # This text is extremely noisy. Parsed from PDFs with random newlines and citations.
        corpus_text += f"{ctx}\n\n====================\n\n"
        
    os.makedirs("./data", exist_ok=True)
    corpus_path = "./data/medical_corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(corpus_text)
        
    print(f"Created {corpus_path} ({os.path.getsize(corpus_path) / 1024 / 1024:.2f} MB of dense medical text)")
    
    # --- STEP 2: INGEST VIA YOUR FASTAPI ENDPOINT ---
    print("\n3. Uploading noisy corpus to your FastAPI ingestion endpoint...")
    with open(corpus_path, "rb") as f:
        files = {"file": ("medical_corpus.txt", f, "text/plain")}
        data = {"tenant_id": TEST_TENANT_ID}
        
        response = requests.post(f"{API_BASE_URL}/documents/upload", files=files, data=data)
        
    if response.status_code != 200:
        print(f"Upload failed: {response.text}")
        return
        
    print(f"Upload Success! {response.json()}")
    print("Waiting 25 seconds for background BM25 index to finish chewing through this dense text...")
    time.sleep(25)
    
    # --- STEP 3: QUERY YOUR FASTAPI ENDPOINT ---
    print("\n4. Querying your API and collecting answers...")
    
    data_for_ragas = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for row in test_cases:
        question = row["question"] # type: ignore
        print(f"\nAsking: {question}")
        
        query_payload = {
            "query": question,
            "tenant_id": TEST_TENANT_ID
        }
        
        res = requests.post(f"{API_BASE_URL}/query/", json=query_payload)
        
        if res.status_code == 200:
            api_data = res.json()
            
            data_for_ragas["question"].append(question)
            data_for_ragas["answer"].append(api_data["answer_markdown"])
            
            retrieved_chunks = [source["full_content"] for source in api_data["sources"]]
            data_for_ragas["contexts"].append(retrieved_chunks)
            
            # PubMed QA format stores the actual detailed answer here
            ground_truth = row["long_answer"] # type: ignore
            data_for_ragas["ground_truth"].append(ground_truth)
        else:
            print(f"Query failed: {res.text}")
    
    # --- STEP 4: EVALUATE YOUR PIPELINE WITH RAGAS ---
    print("\n5. Running RAGAS Evaluation on YOUR API's results...")
    ragas_dataset = Dataset.from_dict(data_for_ragas)
    
    eval_llm = ChatOllama(model="gemma4:31b-cloud", temperature=0)
    eval_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"local_files_only": True} 
    )
    
    result = evaluate(
        dataset=ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=RunConfig(max_workers=2, max_retries=15, timeout=180),
        raise_exceptions=False
    )
    
    print("\n=== FINAL RAGAS SCORES FOR YOUR API ===")
    print(result)
    
    # Save results
    os.makedirs("./data/evals", exist_ok=True)
    df = result.to_pandas() # type: ignore
    df.to_csv("./data/evals/biomed_benchmark.csv", index=False)
    print("\nSaved detailed report to ./data/evals/biomed_benchmark.csv")

if __name__ == "__main__":
    run_end_to_end_test()