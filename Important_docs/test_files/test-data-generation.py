from crewai.tools import BaseTool
import json
from crewai_tools.aws.bedrock.knowledge_base.retriever_tool import BedrockKBRetrieverTool
import boto3
from langchain_aws import BedrockEmbeddings
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from langchain_community.vectorstores import OpenSearchVectorSearch
import os
from typing import Type
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import time
import concurrent.futures

load_dotenv()

# AWS credentials and region
AWS_REGION = "us-east-1"
OPENSEARCH_ENDPOINT = "https://fkakzty7t3bgtpwvdjp3.us-east-1.aoss.amazonaws.com"

session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)
credentials = session.get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    AWS_REGION,
    "aoss",
    session_token=credentials.token
)

# Bedrock client and embeddings
bedrock_client = boto3.client(
    service_name='bedrock-runtime',
    region_name=AWS_REGION,
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

embeddings = BedrockEmbeddings(
    client=bedrock_client,
    region_name=AWS_REGION,
    model_id="amazon.titan-embed-text-v2:0"
)

# Knowledge base retriever
kb_tool = BedrockKBRetrieverTool(
    knowledge_base_id="01ANQMZB9P",
    number_of_results=2,
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_region=AWS_REGION
)

kb_tool2 = BedrockKBRetrieverTool(
    knowledge_base_id="01ANQMZB9P",
    retrieval_configuration = {
  "vectorSearchConfiguration": {
     "numberOfResults": 2, 
     "overrideSearchType": "HYBRID",  
     "filter": {  
             "equals": {"key": "x-amz-bedrock-kb-source-uri", "value": "https://cbre-jira.atlassian.net/wiki/spaces/CLOUD/pages/39605931/Cloud+Platform+Request+Forms"}, 
     
     }
  }
},
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_region=AWS_REGION
)

class RetrieveContentToolInput(BaseModel):
    data: str = Field(..., description="The data to process and extract content from.")

class RetrieveContentTool(BaseTool):
    name: str = "RetrieveContentTool"
    # description: str = "Extracts and returns content from both OpenSearch and Bedrock Knowledge Base."
    # description: str = (
    #     "Extracts and returns content from both OpenSearch and Bedrock Knowledge Base. "
    #     "Tool Arguments: {'query':  'kwargs': {performanceConfig={"latency": "optimized"}}}"
    #     "\n\nExpected input: Pass your query as a string to the 'data' argument. "
    #     "Example: tool_com(data='your question here')\n"
    #     "Do NOT pass a dictionary or use a 'query' key. Only a string is accepted."
    # )

    description: str = (
        "Extracts and returns content from both OpenSearch and Bedrock Knowledge Base. "
        "Tool Arguments: {'query': <string>, 'kwargs': {'performanceConfig': {'latency': 'optimized'}}}"
        "\n\nExpected input: Pass your query as a string to the 'data' argument. "
        "Example: tool_com(data='your question here')\n"
        "Do NOT pass a dictionary or use a 'query' key. Only a string is accepted."
    )

    args_schema: Type[BaseModel] = RetrieveContentToolInput
    
    def _run(self, data: str):
        overall_start = time.time()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_opensearch = executor.submit(self.extract_from_opensearch, data)
            future_kb = executor.submit(self.extract_from_kb, data)
            future_kb2 = executor.submit(self.extract_from_kb2, data)
            opensearch_results = future_opensearch.result()
            kb_content= future_kb.result()
            kb_content2 = future_kb2.result()
        print(f"Total time taken for all operations: {time.time() - overall_start:.2f} seconds")
        
        # Format OpenSearch results
        opensearch_str = "=== ServiceNow Historical Tickets ===\n"
        if opensearch_results == []:
            opensearch_str += "No relevant ServiceNow historical tickets found.\n"
            
        for item in opensearch_results:
            opensearch_str += f"Ticket: {item['Ticket']}\nDescription: {item['Description'][0]}\n---\n"

        # Format KB results
        kb_str = "=== Confluence & SharePoint Results ===\n"
        
        for item in kb_content:
            kb_str += (
                f"Content: {item['content']}\n"
                f"Description: {item['description']}\n"
                f"Link: {item['url']}\n---\n"
            )

        # Format KB2 results
        kb2_str = "=== ServiceNow Reference Links ===\n"
        for item in kb_content2:
            kb2_str += (
                f"Content: {item['content']}\n"
                f"Link: {item['url']}\n---\n"
            )
        if not opensearch_results:
            result_str = f"{kb_str}\n{kb2_str}"
        else:
            result_str = f"{opensearch_str}\n{kb_str}\n{kb2_str}"
        return result_str
        # return {
        #     "Results from opensearch which have all the relevant information of service now historical tickets": opensearch_results,
        #     "Results from confluence which have all the relevant information of confluence and sharepoint data": kb_content,
        #     # "Image data Results from confluence which have all the relevant information of confluence and sharepoint data": kb_desc,
        #     # "Links to reference": kb_urls,
        #     "Service now reference links": kb_content2,
        #     # "Links to reference for ": kb_urls2
        # }

    def extract_from_opensearch(self, query: str):
        start = time.time()
        docs = self._sync_opensearch_search(query)
        print(f"Time for OpenSearch: {time.time() - start:.2f} seconds")
        return [{'Ticket': doc.metadata['ticket_number'], 'Description': [doc.metadata['work_notes']]} for doc in docs]

    def _sync_opensearch_search(self, query: str):
        gh_aoss = OpenSearchVectorSearch(
            index_name="service-now-index",
            embedding_function=embeddings,
            opensearch_url=OPENSEARCH_ENDPOINT,
            http_auth=awsauth,
            timeout=300,
            use_ssl=True,
            connection_class=RequestsHttpConnection,
        )
        return gh_aoss.similarity_search(
            query=query,
            k=2,
            search_type="script_scoring",
            vector_field="embedding",
            text_field="work_notes",
            metadata_field="*",
            score_threshold=0.6
        )

    # def extract_from_kb(self, query: str):
    #     start = time.time()
    #     parsed_data = json.loads(kb_tool.run(query))
    #     content_list = [result["content"] for result in parsed_data["results"] if result['score'] > 0.60] 
    #     desc = [str(result["metadata"].get("x-amz-bedrock-kb-description", "")) for result in parsed_data["results"] if result['score'] > 0.60]
    #     url_list = [result["source_uri"] for result in parsed_data["results"] if result['score'] > 0.60]
    #     answer = "\n".join(content_list), "\n".join(desc), url_list
    #     print(f"Time for Bedrock KB: {time.time() - start:.2f} seconds")
    #     return answer
    
    def extract_from_kb(self, query: str):
        start = time.time()
        parsed_data = json.loads(kb_tool.run(query))
        results = [
            {
                "content": result["content"],
                "description": str(result["metadata"].get("x-amz-bedrock-kb-description", "")),
                "url": result["source_uri"]
            }
            for result in parsed_data["results"]
        ]
        print(f"Time for Bedrock KB: {time.time() - start:.2f} seconds")
        return results

    # def extract_from_kb2(self, query: str):
    #     start = time.time()
    #     query1="get cbre.service-now.com links for" + query
    #     parsed_data = json.loads(kb_tool2.run(query1))
    #     content_list = [result["content"] for result in parsed_data["results"] ]
    #     url_list = [result["source_uri"] for result in parsed_data["results"]]
    #     answer = "\n".join(content_list), url_list
    #     print(f"Time for KB2: {time.time() - start:.2f} seconds")
    #     return answer
    def extract_from_kb2(self, query: str):
        start = time.time()
        query1 = "get cbre.service-now.com links for " + query + " from the Cloud Assistant Bot SNOW Form Links"
        parsed_data = json.loads(kb_tool2.run(query1))
        results = [
            {
                "content": result["content"],
                "url": result["source_uri"]
            }
            for result in parsed_data["results"]
        ]
        print(f"Time for KB2: {time.time() - start:.2f} seconds")
        return results

# A = RetrieveContentToolInput(data="service now ticket aws")
# gh_docs = RetrieveContentTool()._run(A.data)
# print(gh_docs)
import pandas as pd
# import pandas as pd
import requests
# from tool import RetrieveContentTool  

# # Initialize the tool
tool = RetrieveContentTool()

# Define the FastAPI endpoint
url = "http://127.0.0.1:8000/ask"


# Load the Excel file
input_file = "data-test.xlsx"
output_file = "generated_data.xlsx"
df = pd.read_excel(input_file, engine="openpyxl")

# Ensure the 'question' column exists
if 'user_input' not in df.columns:
    raise ValueError("The Excel file must contain a column named 'question'.")

# Prepare columns for responses
df['tool_response'] = ""
df['ai_response'] = ""

# Process each question
for idx, row in df.iterrows():
    print('done')
    question = row['user_input']  # Assuming the column with questions is named 'question'

    # FastAPI call
    data = {
        "question": question,
        "conversation_history": [{"role": "", "content": ""}],
        "regenerate": False
    }

    try:
        response = requests.post(url, json=data)
        df.at[idx, 'ai_response'] = response.json().get("answer") if response.status_code == 200 else f"Error: {response.status_code}"
    except Exception as e:
        df.at[idx, 'ai_response'] = f"Exception: {str(e)}"

    # Call the RetrieveContentTool
    try:
        tool_result = tool._run(data=question)
        df.at[idx, 'tool_response'] = str(tool_result)
        # print(f"Tool response for question {idx}: {tool_result}")
    except Exception as e:
        df.at[idx, 'tool_response'] = f"Tool Exception: {str(e)}"

# Save the updated Excel file
df.to_excel(output_file, index=False, engine="openpyxl")
print(f"Responses saved to {output_file}")