import os
import json
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine import RAGEngine
from src.llm_client import LLMClient
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GPTModel
import time

load_dotenv()

def run_evaluation():
    rag_engine = RAGEngine()
    if not rag_engine.load_index():
        print("Error: Index not found.")
        return
        
    llm_client = LLMClient()
    
    groq_judge = GPTModel(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

    with open("tests/test_cases.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    faithfulness = FaithfulnessMetric(threshold=0.7, model=groq_judge, async_mode=False)
    relevancy = AnswerRelevancyMetric(threshold=0.7, model=groq_judge, async_mode=False)
    policy_recency_metric = GEval(
        name="Policy Recency & Conflict Resolution",
        model=groq_judge,
        criteria=(
            "1. If a 2025 update exists, AI must prioritize it. "
            "2. If no 2025 update exists, AI is correct to use 2024 data as the final rule. "
            "3. Do not penalize for missing 2025 data if it is not present in the Retrieval Context."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
        threshold=0.7,
        async_mode=False
    )

    print(f"Starting Groq Evaluation for {len(data)} cases...")

    for i, item in enumerate(data):
        print(f"\nEvaluating Case {i+1}: {item['input']}")
        
        expanded_query = llm_client.expand_query(item["input"])
        retrieved_docs = rag_engine.search(expanded_query)
        context = [doc.page_content for doc in retrieved_docs]
        actual_answer = llm_client.generate_answer(item["input"], retrieved_docs)

        test_case = LLMTestCase(
            input=item["input"],
            actual_output=actual_answer,
            expected_output=item["expected_output"],
            retrieval_context=context
        )
        
        evaluate([test_case], metrics=[faithfulness, relevancy, policy_recency_metric])

        if i < len(data) - 1:
            print("Waiting 10s for TPM reset...")
            time.sleep(10)

if __name__ == "__main__":
    run_evaluation()