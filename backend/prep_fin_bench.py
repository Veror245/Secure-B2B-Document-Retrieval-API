import os
import random
from datasets import load_dataset

def extract_evidence(row):
    """Safely extract evidence text regardless of dataset schema variations."""
    if 'evidence_text' in row:
        return str(row['evidence_text'])
    elif 'evidence' in row:
        evidence = row['evidence']
        # In FinanceBench, evidence is often a list of dictionaries
        if isinstance(evidence, list):
            texts = []
            for e in evidence:
                if isinstance(e, dict) and 'evidence_text' in e:
                    texts.append(str(e['evidence_text']))
                elif isinstance(e, str):
                    texts.append(e)
            return "\n".join(texts)
        return str(evidence)
    elif 'context' in row:
        return str(row['context'])
    return ""

def prepare_finance_benchmark():
    print("Downloading PatronusAI/financebench...")
    # This dataset contains tough financial questions and exact context chunks
    dataset = load_dataset("PatronusAI/financebench", split="train")
    
    # Let's dynamically get the total number of rows available
    total_rows = len(dataset)
    print(f"Dataset loaded with {total_rows} rows.")
    
    # Let's take 25 questions for our actual test
    test_cases = dataset.select(range(25))
    
    # Let's take all remaining chunks to act as "distractors" (the haystack)
    # This forces ChromaDB and BM25 to actually work hard to find the right chunk
    print(f"Building the haystack (injecting {total_rows - 25} distractor documents)...")
    distractor_cases = dataset.select(range(25, total_rows))
    
    corpus_text = "# FinanceBench Enterprise Knowledge Base\n\n"
    
    # 1. Add all the distractor chunks
    for row in distractor_cases:
        text = extract_evidence(row)
        if text.strip():
            corpus_text += f"{text}\n\n---\n\n"
        
    # 2. Add the actual chunks we need to find, shuffled in
    for row in test_cases:
        text = extract_evidence(row)
        if text.strip():
            corpus_text += f"{text}\n\n---\n\n"
        
    # Save the massive, complex financial corpus
    os.makedirs("./data", exist_ok=True)
    corpus_path = "./data/financebench_corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(corpus_text)
        
    print(f"\nCreated {corpus_path}")
    print(f"Total size: {os.path.getsize(corpus_path) / 1024 / 1024:.2f} MB")
    print("\nNext Steps:")
    print("1. Upload this file to your /documents/upload endpoint.")
    print("2. Wait for the BM25 index to finish building.")
    print("3. Run your RAGAS eval script against these 25 questions!")

if __name__ == "__main__":
    prepare_finance_benchmark()