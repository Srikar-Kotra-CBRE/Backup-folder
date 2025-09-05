# import boto3
# import json
# from typing import List
# from langchain.embeddings.base import Embeddings
# from ragas import evaluate
# from ragas.metrics import (
#    answer_correctness,
#    answer_relevancy,
#    context_precision,
#    context_recall,
#    faithfulness
# )
# # Custom Bedrock Embedding class
# class BedrockTitanEmbeddings(Embeddings):
#    def __init__(self, region_name="us-east-1"):
#        self.bedrock = boto3.client("bedrock-runtime", region_name=region_name)
#        self.model_id = "amazon.titan-embed-text-v1"
#    def embed_documents(self, texts: List[str]) -> List[List[float]]:
#        return [self.embed_query(t) for t in texts]
#    def embed_query(self, text: str) -> List[float]:
#        payload = { "inputText": text }
#        response = self.bedrock.invoke_model(
#            body=json.dumps(payload),
#            modelId=self.model_id,
#            accept="application/json",
#            contentType="application/json"
#        )
#        body = json.loads(response['body'].read())
#        return body["embedding"]
# # Wrap for RAGAS
# from ragas.embeddings import LangchainEmbeddingsWrapper
# embedding = LangchainEmbeddingsWrapper(BedrockTitanEmbeddings())
# #input csv file and then pass that to the end point \ask then add to the csv file and then give this library and config. Apect critc.
# from ragas.context import get_context
# # Define sample QA data
# qa_data = [
#    {
#        "question": "Who wrote Pride and Prejudice?",
#        "answer": "Jane Austen wrote Pride and Prejudice.",
#        "ground_truth": "Jane Austen",
#        "contexts": [
#            "Jane Austen was an English novelist known for books such as Pride and Prejudice.",
#            "Pride and Prejudice is a romantic novel published in 1813."
#        ]
#    },
#    {
#        "question": "What is the capital of France?",
#        "answer": "The capital city of France is Paris.",
#        "ground_truth": "Paris",
#        "contexts": [
#            "Paris is the capital and most populous city of France.",
#            "France is located in Western Europe."
#        ]
#    }
# ]
# # Evaluate with RAGAS
# # from ragas.config import set_embeddings
# # set_embeddings(embedding)  # Set the embedding for RAGAS
# results = evaluate(
#    dataset=qa_data,  # List[dict] format
#    metrics=[
#        answer_correctness,
#        answer_relevancy,
#        context_precision,
#        context_recall,
#        faithfulness
#    ],
# #    embedding=embedding
# )
# # Print results
# print("📊 RAGAS Evaluation Results (Titan + Bedrock):")
# for metric, score in results.items():
#    print(f"{metric}: {score:.3f}")