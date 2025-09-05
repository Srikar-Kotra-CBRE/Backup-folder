import json
import boto3
import os
import logging
import datetime
import traceback
import time
import concurrent.futures
from typing import List, Dict, Any
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
ssm = boto3.client('ssm')
dynamodb = boto3.resource('dynamodb')
ses = boto3.client('ses')

# Constants
USER_MANAGER_TABLE = 'user_manager_map'
DOC_REVIEW_STATUS_TABLE = 'doc_review_status'
PARAMETER_PATH = '/document-review/'
AWS_REGION = "us-east-1"

# Cache for parameters
param_cache = {}

def get_parameter(param_name):
    """
    Get parameter from Parameter Store with caching
    """
    if param_name in param_cache:
        return param_cache[param_name]
    
    try:
        response = ssm.get_parameter(Name=f"{PARAMETER_PATH}{param_name}")
        param_value = response['Parameter']['Value']
        param_cache[param_name] = param_value
        return param_value
    except Exception as e:
        logger.error(f"Error retrieving parameter {param_name}: {str(e)}")
        raise

def get_kb_id():
    """Get knowledge base ID from Parameter Store"""
    return get_parameter('KNOWLEDGE_BASE_ID')

def get_data_sources():
    """Get data source names from Parameter Store"""
    data_sources_json = get_parameter('DATA_SOURCES')
    return json.loads(data_sources_json)

class BedrockKnowledgeBaseRetriever:
    """Class to retrieve documents from Bedrock Knowledge Base"""
    
    def __init__(self, knowledge_base_id, aws_region=AWS_REGION):
        # Create a Bedrock agent runtime client for knowledge bases
        self.bedrock_agent_client = boto3.client(
            service_name='bedrock-agent-runtime',
            region_name=aws_region
        )
        
        # Store the knowledge base ID
        self.knowledge_base_id = knowledge_base_id
        
    def retrieve_documents(self, data_source=None, max_results=25):
        """
        Retrieve documents from the knowledge base
        
        Args:
            data_source: Optional data source filter
            max_results: Maximum number of results to return
            
        Returns:
            List of document results
        """
        try:
            # Build the retrieve request
            retrieve_params = {
                'knowledgeBaseId': self.knowledge_base_id,
                'retrievalQuery': {
                    'text': f"data_source:{data_source}" if data_source else ""
                }
            }
            
            logger.info(f"Querying knowledge base with params: {retrieve_params}")
            
            # Call the retrieve API
            response = self.bedrock_agent_client.retrieve(**retrieve_params)
            
            # Extract documents from the response
            documents = []
            
            for result in response.get('retrievalResults', []):
                # Extract content text
                content_text = ""
                if 'content' in result and 'text' in result['content']:
                    content_text = result['content']['text']
                
                # Create document object
                doc = {
                    'document_id': '',
                    'title': 'Unknown Document',
                    'url': '',
                    'content': content_text,
                    'data_source': data_source or ''
                }
                
                # Extract metadata
                if 'documentMetadata' in result:
                    metadata = result['documentMetadata']
                    
                    if 'sourceAttributes' in metadata and metadata['sourceAttributes']:
                        source_attrs = metadata['sourceAttributes']
                        
                        if 'title' in source_attrs:
                            doc['title'] = source_attrs['title']
                        
                        if 'location' in source_attrs:
                            doc['url'] = source_attrs['location']
                            # Use the filename as document_id
                            if '/' in doc['url']:
                                doc['document_id'] = doc['url'].split('/')[-1]
                            else:
                                doc['document_id'] = doc['url']
                
                # Add the document to the list
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents from knowledge base")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving from knowledge base: {str(e)}")
            logger.error(traceback.format_exc())
            return []

def fetch_all_documents():
    """
    Fetch all documents from all data sources in the knowledge base
    """
    kb_id = get_kb_id()
    data_sources = get_data_sources()
    all_documents = []
    
    # Create the retriever
    retriever = BedrockKnowledgeBaseRetriever(kb_id)
    
    # Use parallel processing for faster retrieval
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a future for each data source
        future_to_data_source = {
            executor.submit(retriever.retrieve_documents, data_source): data_source
            for data_source in data_sources
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_data_source):
            data_source = future_to_data_source[future]
            try:
                documents = future.result()
                logger.info(f"Retrieved {len(documents)} documents from {data_source}")
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Error retrieving from {data_source}: {str(e)}")
    
    logger.info(f"Total documents retrieved: {len(all_documents)}")
    return all_documents

def extract_review_info_with_claude(documents: List[Dict[str, Any]]):
    """
    Use Claude to extract review information from document content
    """
    # Initialize Bedrock Runtime client for Claude
    bedrock_runtime = boto3.client('bedrock-runtime', region_name=AWS_REGION)
    
    documents_with_review_info = []
    batch_size = 5  # Process documents in batches to avoid large prompts
    
    # Process documents in batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        
        # Create a prompt for Claude to extract review information
        prompt = "For each document below, extract the review information including 'Updated by', 'Reviewed by', 'Updated date', and 'Next review date' if available. Also indicate if there is no review block present. Return the results in JSON format.\n\n"
        
        for idx, doc in enumerate(batch):
            # Limit content length to avoid token limits
            content = doc.get('content', '')
            if content and len(content) > 2000:
                content = content[:2000] + "..."
                
            prompt += f"Document {idx+1}: {doc.get('title', 'Unknown Document')}\nContent: {content}\n\n"
        
        try:
            # Invoke Claude
            response = bedrock_runtime.invoke_model(
                modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 500,
                    "temperature": 0,
                    "system": "You are an assistant that extracts document review information. For each document, check if there is a review block containing 'Updated by', 'Reviewed by', 'Updated date', and 'Next review date'. Return a JSON array where each item has the fields: document_index, has_review_block (true/false), updated_by, reviewed_by, updated_date, next_review_date.",
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                })
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            
            if 'content' in response_body and len(response_body['content']) > 0:
                content_text = response_body['content'][0]['text']
                
                # Extract JSON from the response
                try:
                    start_idx = content_text.find('[')
                    end_idx = content_text.rfind(']') + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = content_text[start_idx:end_idx]
                        review_info = json.loads(json_str)
                        
                        # Match review info with the original documents
                        for j, info in enumerate(review_info):
                            if j < len(batch):
                                # Add review info to the original document
                                doc = batch[j].copy()
                                doc['has_review_block'] = info.get('has_review_block', False)
                                doc['updated_by'] = info.get('updated_by')
                                doc['reviewed_by'] = info.get('reviewed_by')
                                doc['updated_date'] = info.get('updated_date')
                                doc['next_review_date'] = info.get('next_review_date')
                                
                                documents_with_review_info.append(doc)
                    else:
                        # If JSON parsing fails, add the batch without review info
                        for doc in batch:
                            doc_copy = doc.copy()
                            doc_copy['has_review_block'] = False
                            documents_with_review_info.append(doc_copy)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON from model response")
                    # Add the batch without review info
                    for doc in batch:
                        doc_copy = doc.copy()
                        doc_copy['has_review_block'] = False
                        documents_with_review_info.append(doc_copy)
            else:
                # Add the batch without review info if no content in response
                for doc in batch:
                    doc_copy = doc.copy()
                    doc_copy['has_review_block'] = False
                    documents_with_review_info.append(doc_copy)
                    
        except Exception as e:
            logger.error(f"Error extracting review info: {str(e)}")
            logger.error(traceback.format_exc())
            # Add the batch without review info
            for doc in batch:
                doc_copy = doc.copy()
                doc_copy['has_review_block'] = False
                documents_with_review_info.append(doc_copy)
    
    return documents_with_review_info

def determine_document_status(documents_with_review_info):
    """
    Determine which documents need review based on review dates
    """
    today = datetime.datetime.now()
    review_threshold_days = int(get_parameter('REVIEW_THRESHOLD_DAYS'))
    threshold_date = today + datetime.timedelta(days=review_threshold_days)
    
    flagged_documents = []
    
    for doc in documents_with_review_info:
        issue = None
        
        # Check if the document has a review block
        if not doc.get('has_review_block', False):
            issue = "Missing review block"
        
        # Check if the document has a next review date
        elif doc.get('next_review_date'):
            try:
                # Try to parse the date - handle different formats
                next_review_date = None
                date_formats = ['%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y']
                
                for date_format in date_formats:
                    try:
                        next_review_date = datetime.datetime.strptime(doc['next_review_date'], date_format)
                        break
                    except ValueError:
                        continue
                
                if next_review_date:
                    if next_review_date < today:
                        issue = "Review date expired"
                    elif next_review_date <= threshold_date:
                        issue = "Review due within 60 days"
                else:
                    issue = "Invalid review date format"
            except Exception:
                issue = "Invalid review date format"
        else:
            issue = "Missing next review date"
        
        # If the document needs attention, add it to the flagged list
        if issue:
            flagged_doc = {
                'document_id': doc.get('document_id', doc.get('url', '')),
                'title': doc.get('title', 'Unknown Document'),
                'url': doc.get('url', ''),
                'data_source': doc.get('data_source', ''),
                'issue': issue,
                'has_review_block': doc.get('has_review_block', False),
                'updated_by': doc.get('updated_by'),
                'reviewed_by': doc.get('reviewed_by'),
                'updated_date': doc.get('updated_date'),
                'next_review_date': doc.get('next_review_date'),
                'assignment_status': 'Pending',
                'notification_count': 0,
                'last_notification_date': None
            }
            flagged_documents.append(flagged_doc)
    
    return flagged_documents

def lambda_handler(event, context):
    """
    Lambda handler function
    """
    try:
        logger.info(f"Starting document review process at {datetime.datetime.now()}")
        
        # Step 1: Fetch all documents from the knowledge base
        start_time = time.time()
        documents = fetch_all_documents()
        logger.info(f"Fetched {len(documents)} documents in {time.time() - start_time:.2f} seconds")
        
        if not documents:
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'No documents retrieved from knowledge base.',
                    'sample_flagged_documents': []
                })
            }
        
        # Step 2: Extract review information from documents
        start_time = time.time()
        documents_with_review_info = extract_review_info_with_claude(documents)
        logger.info(f"Extracted review info for {len(documents_with_review_info)} documents in {time.time() - start_time:.2f} seconds")
        
        # Step 3: Determine which documents need review
        start_time = time.time()
        flagged_documents = determine_document_status(documents_with_review_info)
        logger.info(f"Flagged {len(flagged_documents)} documents in {time.time() - start_time:.2f} seconds")
        
        # For testing purposes, return the first few flagged documents
        # sample_docs = flagged_documents[:5] if len(flagged_documents) > 5 else flagged_documents
        # Return all flagged documents instead of a sample
        sample_docs = flagged_documents
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Successfully processed {len(documents)} documents and flagged {len(flagged_documents)} for review',
                'flagged_documents': sample_docs
            }, default=str)
        }
    except Exception as e:
        logger.error(f"Error in document review process: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }