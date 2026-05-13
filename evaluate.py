import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings as LangchainOpenAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall
from ragas.run_config import RunConfig
from datasets import Dataset
from ingest import load_saved_documents
from rag import ask_with_sources

load_dotenv()

TESTSET_PATH = "testset.csv"
RESULTS_PATH = "evaluation_results.csv"

# ── Generate testset ──────────────────────────────────────────────────────────

def generate_testset():
    print("Loading documents...")
    all_documents = load_saved_documents()
    print(f"Using {len(all_documents)} pages total")
    print("Generating test questions and answers...")

    openai_client = OpenAI()
    run_config = RunConfig(
        max_workers=1,
        max_retries=5,
        timeout=120
    )

    generator_llm = llm_factory("gpt-4o-mini", client=openai_client)
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=openai_client)

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=embeddings
    )
    generator.run_config = run_config

    for attempt in range(3):
        try:
            testset = generator.generate_with_langchain_docs(
                all_documents,
                testset_size=30
            )
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                print("Retrying in 15 seconds...")
                time.sleep(15)
            else:
                raise

    df = testset.to_pandas()

    # Keep only simple single-hop questions
    if "synthesizer_name" in df.columns:
        simple = df[df["synthesizer_name"] == "single_hop_specific_query_synthesizer"]
        if len(simple) > 0:
            print(f"Filtered to {len(simple)} simple questions from {len(df)} total")
            df = simple

    df = df.head(15)
    df.to_csv(TESTSET_PATH, index=False)
    print(f"\nSaved {len(df)} questions to {TESTSET_PATH}")
    print(df[["user_input", "reference"]].to_string())

# ── Run evaluation ────────────────────────────────────────────────────────────

def run_evaluation():
    if not os.path.exists(TESTSET_PATH):
        print("testset.csv not found. Run generate_testset() first.")
        return

    df = pd.read_csv(TESTSET_PATH)

    results = []
    print("Running RAG on test questions...")

    for i, row in df.iterrows():
        question = row["user_input"]
        ground_truth = row["reference"]

        print(f"  Question {i+1}/{len(df)}: {question[:60]}...")
        answer, sources = ask_with_sources(question)

        results.append({
            "user_input": question,
            "response": answer,
            "reference": ground_truth,
            "retrieved_contexts": sources
        })

    dataset = Dataset.from_list(results)

    openai_client = OpenAI()
    llm = llm_factory("gpt-4o-mini", client=openai_client)
    embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=openai_client
    )

    print("\nEvaluating with RAGAS...")
    scores = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            ContextPrecision(),
            ContextRecall()
        ],
        llm=llm,
        embeddings=embeddings
    )

    print("\n--- RAGAS SCORES ---")
    print(scores)

    scores_df = scores.to_pandas()
    scores_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nDetailed results saved to {RESULTS_PATH}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_testset()
    run_evaluation()