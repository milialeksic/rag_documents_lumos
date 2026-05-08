import os
from dotenv import load_dotenv
from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ingest import load_saved_documents

load_dotenv()


from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

def generate_testset():
    print("Loading documents...")
    documents = load_saved_documents()
    
    print("Generating test questions and answers...")
    
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    generator = TestsetGenerator(llm=generator_llm, embedding_model=embeddings)
    
    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=15
    )
    
    df = testset.to_pandas()
    df.to_csv("testset.csv", index=False)
    print(f"Generated {len(df)} test questions saved to testset.csv")
    print(df[["user_input", "reference"]].to_string())
    
    return testset


from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from rag import ask_with_sources
import pandas as pd

def run_evaluation():
    df = pd.read_csv("testset.csv")
    
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
    
    # Wrap LLM and embeddings for RAGAS
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())
    
    print("\nEvaluating with RAGAS...")
    scores = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=llm,
        embeddings=embeddings
    )
    
    print("\n--- RAGAS SCORES ---")
    print(scores)
    
    scores_df = scores.to_pandas()
    scores_df.to_csv("evaluation_results.csv", index=False)
    print("\nDetailed results saved to evaluation_results.csv")

if __name__ == "__main__":
    # generate_testset()
    run_evaluation()