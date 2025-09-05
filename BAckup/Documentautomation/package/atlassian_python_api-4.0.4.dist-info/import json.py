import json
import boto3
import logging
import os
import time
from datetime import datetime, timedelta
import concurrent.futures

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
ssm = boto3.client('ssm')
dynamodb = boto3.resource('dynamodb')
bedrock_runtime = boto3.client('bedrock-agent-runtime', region_name="us-east-1")
bedrock_agent = boto3.client('bedrock-runtime', region_name="us-east-1")  # For knowledge base queries

def get_parameter(name):
    """Retrieve a parameter from SSM Parameter Store"""
    try:
        response = ssm.get_parameter(Name=name)
        return response['Parameter']['Value']
    except Exception as e:
        logger.error(f"Error retrieving parameter {name}: {str(e)}")
        raise

def get_parameters_by_path(path):
    """Retrieve all parameters under a path from SSM Parameter Store"""
    try:
        response = ssm.get_parameters_by_path(Path=path, Recursive=True)
        return {p['Name'].split('/')[-1]: p['Value'] for p in response['Parameters']}
    except Exception as e:
        logger.error(f"Error retrieving parameters from {path}: {str(e)}")
        raise

def extract_review_info_with_claude(document_content, document_title):
    """
    Use Claude to extract review information from document content
    """
    try:
        # Prepare prompt for Claude
        prompt = f"""
        Extract review information from the following document:
        
        TITLE: {document_title}
        
        CONTENT:
        {document_content[:20000]}  # Limit content to avoid token limits
        
        Look for a review block/section that contains information like "Updated by", "Reviewed by", "Updated date", 
        "Next review date", or similar variations.
        
        Respond in JSON format only with these fields:
        - has_review_block: Boolean indicating if a review block was found
        - updated_by: Name or email of who updated the document (null if not found)
        - reviewed_by: Name or email of who reviewed the document (null if not found)
        - updated_date: Date when document was updated in YYYY-MM-DD format (null if not found)
        - next_review_date: Date when document should be reviewed next in YYYY-MM-DD format (null if not found)
        - confidence: "high", "medium", or "low" indicating confidence in the extraction
        
        Example response:
        {{
          "has_review_block": true,
          "updated_by": "john.smith@example.com",
          "reviewed_by": "jane.doe@example.com",
          "updated_date": "2023-05-15",
          "next_review_date": "2024-05-15",
          "confidence": "high"
        }}
        
        If no review information is found, return:
        {{
          "has_review_block": false,
          "updated_by": null,
          "reviewed_by": null,
          "updated_date": null,
          "next_review_date": null,
          "confidence": "high"
        }}
        """

        # Invoke Claude via Bedrock
        model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"  # Use appropriate model
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 5000,
                "temperature": 0,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
        )
        
        # Parse response
        response_body = json.loads(response['body'].read().decode('utf-8'))
        ai_response = response_body['content'][0]['text']
        
        # Extract JSON from response
        try:
            # Find JSON in the response (might be surrounded by markdown code block)
            if "```json" in ai_response:
                json_text = ai_response.split("```json")[1].split("```")[0].strip()
            elif "```" in ai_response:
                json_text = ai_response.split("```")[1].strip()
            else:
                json_text = ai_response.strip()
                
            extracted_data = json.loads(json_text)
            
            # Validate structure
            required_fields = ['has_review_block', 'updated_by', 'reviewed_by', 
                              'updated_date', 'next_review_date', 'confidence']
            for field in required_fields:
                if field not in extracted_data:
                    extracted_data[field] = None
                    
            return extracted_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude's response as JSON: {str(e)}")
            logger.error(f"Claude response: {ai_response}")
            return {
                "has_review_block": False,
                "updated_by": None,
                "reviewed_by": None,
                "updated_date": None,
                "next_review_date": None,
                "confidence": "low"
            }
            
    except Exception as e:
        logger.error(f"Error extracting review info with Claude: {str(e)}")
        return {
            "has_review_block": False,
            "updated_by": None,
            "reviewed_by": None,
            "updated_date": None,
            "next_review_date": None,
            "confidence": "low"
        }

def fetch_documents_from_data_source(knowledge_base_id, data_source):
    """
    Fetch documents from a specific data source in the knowledge base
    """
    documents = []
    next_token = None
    
    try:
        # Paginate through results
        while True:
            kwargs = {
                'knowledgeBaseId': knowledge_base_id,
                'retrievalQuery': {
                    'text': '*'  # Retrieve all documents
                },
                'retrievalConfiguration': {
                    'vectorSearchConfiguration': {
                        'numberOfResults': 100  # Maximum allowed per call
                    }
                }
            }
            
            # Add filter for specific data source
            if data_source:
                kwargs['retrievalConfiguration']['documentFilterConfiguration'] = {
                    'filters': [
                        {
                            'key': 'dataSourceId',
                            'value': data_source
                        }
                    ]
                }
                
            # Add pagination token if we have one
            if next_token:
                kwargs['nextToken'] = next_token
                
            # Make the API call
            response = bedrock_runtime.retrieve(**kwargs)
            
            # Process the returned documents
            for result in response.get('retrievalResults', []):
                document = {
                    'doc_id': result.get('location', {}).get('s3Location', {}).get('uri', 'unknown'),
                    'title': result.get('metadata', {}).get('title', 'Unknown Title'),
                    'url': result.get('metadata', {}).get('source', 'Unknown URL'),
                    'source_type': data_source,
                    'content': result.get('content', ''),
                    'metadata': result.get('metadata', {})
                }
                documents.append(document)
                
            # Check if there are more results
            next_token = response.get('nextToken')
            if not next_token:
                break
                
        logger.info(f"Fetched {len(documents)} documents from data source: {data_source}")
        return documents
        
    except Exception as e:
        logger.error(f"Error fetching documents from data source {data_source}: {str(e)}")
        return []

def process_document(document):
    """
    Process a single document to extract review info and determine status
    """
    try:
        # Extract review info using Claude
        review_info = extract_review_info_with_claude(document['content'], document['title'])
        
        # Add review info to document
        document['has_review_block'] = review_info['has_review_block']
        document['updated_by'] = review_info['updated_by']
        document['reviewed_by'] = review_info['reviewed_by']
        document['updated_date'] = review_info['updated_date']
        document['next_review_date'] = review_info['next_review_date']
        document['extraction_confidence'] = review_info['confidence']
        
        # Determine document status
        document['status'] = determine_document_status(review_info)
        
        # Remove content to save storage space
        if 'content' in document:
            del document['content']
            
        return document
        
    except Exception as e:
        logger.error(f"Error processing document {document.get('title', 'Unknown')}: {str(e)}")
        document['status'] = 'ERROR'
        document['issue'] = str(e)
        
        # Remove content to save storage space
        if 'content' in document:
            del document['content']
            
        return document

def determine_document_status(review_info):
    """
    Determine the status of a document based on review info
    """
    try:
        # Get review threshold
        review_threshold_days = int(get_parameter('/document-review/REVIEW_THRESHOLD_DAYS'))
        today = datetime.now().date()
        threshold_date = today + timedelta(days=review_threshold_days)
        
        if not review_info['has_review_block']:
            return {
                'code': 'MISSING_REVIEW_BLOCK',
                'issue': 'Document is missing review information block'
            }
            
        if not review_info['next_review_date']:
            return {
                'code': 'MISSING_REVIEW_DATE',
                'issue': 'Document is missing next review date'
            }
            
        # Parse next review date
        try:
            next_review_date = datetime.strptime(review_info['next_review_date'], '%Y-%m-%d').date()
            
            if next_review_date <= today:
                return {
                    'code': 'EXPIRED',
                    'issue': f'Document review date has expired (due on {next_review_date.strftime("%Y-%m-%d")})'
                }
                
            if next_review_date <= threshold_date:
                return {
                    'code': 'DUE_SOON',
                    'issue': f'Document review is due soon (due on {next_review_date.strftime("%Y-%m-%d")})'
                }
                
            # Document is OK
            return {
                'code': 'COMPLIANT',
                'issue': None
            }
            
        except (ValueError, TypeError) as e:
            return {
                'code': 'INVALID_DATE_FORMAT',
                'issue': f'Invalid date format for next review date: {review_info["next_review_date"]}'
            }
            
    except Exception as e:
        logger.error(f"Error determining document status: {str(e)}")
        return {
            'code': 'ERROR',
            'issue': f'Error determining status: {str(e)}'
        }

def save_document_to_dynamodb(document):
    """
    Save document to DynamoDB doc_review_status table
    """
    try:
        table = dynamodb.Table('doc_review_status')
        
        # Prepare item for DynamoDB
        item = {
            'doc_id': document['doc_id'],
            'title': document['title'],
            'url': document['url'],
            'source_type': document['source_type'],
            'has_review_block': document.get('has_review_block', False),
            'updated_by': document.get('updated_by'),
            'reviewed_by': document.get('reviewed_by'),
            'updated_date': document.get('updated_date'),
            'next_review_date': document.get('next_review_date'),
            'extraction_confidence': document.get('extraction_confidence'),
            'status_code': document['status']['code'],
            'issue': document['status']['issue'],
            'last_checked': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'assigned_to': None,
            'manager': None,
            'notification_count': 0,
            'last_notification_date': None
        }
        
        # Only save documents that need attention (not COMPLIANT)
        if document['status']['code'] != 'COMPLIANT':
            table.put_item(Item=item)
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error saving document to DynamoDB: {str(e)}")
        return False

def lambda_handler(event, context):
    """
    Main Lambda function handler
    """
    try:
        start_time = time.time()
        logger.info("Starting document review automation process")
        
        # Get configuration from Parameter Store
        kb_id = get_parameter('/document-review/KNOWLEDGE_BASE_ID')
        data_sources_json = get_parameter('/document-review/DATA_SOURCES')
        data_sources = json.loads(data_sources_json)
        
        logger.info(f"Using knowledge base: {kb_id}")
        logger.info(f"Processing data sources: {data_sources}")
        
        # Retrieve documents from each data source in parallel
        all_documents = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(data_sources)) as executor:
            future_to_data_source = {
                executor.submit(fetch_documents_from_data_source, kb_id, data_source): data_source
                for data_source in data_sources
            }
            
            for future in concurrent.futures.as_completed(future_to_data_source):
                data_source = future_to_data_source[future]
                try:
                    documents = future.result()
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"Error fetching from data source {data_source}: {str(e)}")
        
        logger.info(f"Retrieved {len(all_documents)} documents in total")
        
        # Process documents in parallel (process review info and determine status)
        processed_documents = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_document = {
                executor.submit(process_document, document): document
                for document in all_documents
            }
            
            for future in concurrent.futures.as_completed(future_to_document):
                document = future_to_document[future]
                try:
                    processed_doc = future.result()
                    processed_documents.append(processed_doc)
                except Exception as e:
                    logger.error(f"Error processing document: {str(e)}")
        
        logger.info(f"Processed {len(processed_documents)} documents")
        
        # Save flagged documents to DynamoDB
        flagged_documents = []
        saved_count = 0
        for doc in processed_documents:
            if doc['status']['code'] != 'COMPLIANT':
                flagged_documents.append(doc)
                if save_document_to_dynamodb(doc):
                    saved_count += 1
        
        logger.info(f"Identified {len(flagged_documents)} flagged documents")
        logger.info(f"Saved {saved_count} documents to DynamoDB")
        
        # Prepare summary
        status_counts = {}
        for doc in processed_documents:
            status = doc['status']['code']
            if status in status_counts:
                status_counts[status] += 1
            else:
                status_counts[status] = 1
        
        execution_time = time.time() - start_time
        
        # Return summary
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Document review scanning complete',
                'total_documents': len(processed_documents),
                'flagged_documents': len(flagged_documents),
                'status_counts': status_counts,
                'execution_time_seconds': execution_time,
                'sample_flagged': [doc for doc in flagged_documents[:5]]  # Include first 5 flagged docs as sample
            })
        }
        
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Error processing documents',
                'error': str(e)
            })
        }