import json
import boto3
import os
import logging
import datetime
import traceback
import re
import uuid
import hashlib
from decimal import Decimal
from boto3.dynamodb.conditions import Key
from atlassian import Confluence

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
bedrock_runtime = boto3.client('bedrock-runtime', region_name="us-east-1")

# Read config from environment variables for Lambda compatibility
CONFLUENCE_URL = os.environ.get('CONFLUENCE_URL', 'https://cbre-jira.atlassian.net/wiki')
CONFLUENCE_USERNAME = os.environ.get('CONFLUENCE_USERNAME', 'your-email@domain.com')
CONFLUENCE_API_TOKEN = os.environ.get('CONFLUENCE_API_TOKEN', 'your-api-token')
CONFLUENCE_SPACE_KEY = os.environ.get('CONFLUENCE_SPACE_KEY', 'CLOUD')
CONFLUENCE_HOME_PAGE_ID = os.environ.get('CONFLUENCE_HOME_PAGE_ID', '39583749')  # Set to actual home page ID for CLOUD space
DOC_REVIEW_TABLE_NAME = os.environ.get('DOC_REVIEW_TABLE_NAME', 'doc_review_status')
USER_MANAGER_TABLE_NAME = os.environ.get('USER_MANAGER_TABLE_NAME', 'user_manager_map')

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')
ssm = boto3.client('ssm')

# DynamoDB tables (configurable)
doc_review_table = dynamodb.Table(DOC_REVIEW_TABLE_NAME)
user_manager_table = dynamodb.Table(USER_MANAGER_TABLE_NAME)

def get_parameter(parameter_name, default_value=None):
    """Get parameter from AWS Systems Manager Parameter Store"""
    try:
        response = ssm.get_parameter(Name=parameter_name, WithDecryption=True)
        return response['Parameter']['Value']
    except Exception as e:
        logger.warning(f"Could not get parameter {parameter_name}: {str(e)}")
        if default_value is not None:
            logger.info(f"Using default value for {parameter_name}: {default_value}")
        return default_value

# --- NEW: Confluence scraping logic ---
def retrieve_all_confluence_documents():
    """
    Retrieve all pages from the Confluence space (CLOUD) using the space key, not just descendants of a home page.
    """
    try:
        confluence = Confluence(
            url=CONFLUENCE_URL,
            username=CONFLUENCE_USERNAME,
            password=CONFLUENCE_API_TOKEN
        )
        # Get all pages in the space (tree under /wiki/spaces/CLOUD/pages)
        pages = confluence.get_all_pages_from_space(CONFLUENCE_SPACE_KEY, start=0, limit=5000, status=None, expand='body.storage,version,history')
        logger.info(f"Found {len(pages)} pages in {CONFLUENCE_SPACE_KEY} space")
        all_documents = []
        for page in pages:
            try:
                page_id = page['id']
                title = page['title']
                url = f"{CONFLUENCE_URL}/spaces/{CONFLUENCE_SPACE_KEY}/pages/{page_id}/{title.replace(' ', '+')}"
                # Get content and metadata (already expanded)
                content = page.get('body', {}).get('storage', {}).get('value', '')
                author = page.get('history', {}).get('createdBy', {}).get('displayName', '')
                last_modified = page.get('version', {}).get('when', '')
                doc = {
                    'document_id': page_id,
                    'title': title,
                    'url': url,
                    'content': content,
                    'data_source': 'confluence',
                    'author': author,
                    'last_modified': last_modified
                }
                all_documents.append(doc)
            except Exception as e:
                logger.warning(f"Error processing page {page.get('title', 'Unknown')}: {str(e)}")
                continue
        logger.info(f"Total unique documents retrieved from Confluence: {len(all_documents)}")
        return all_documents
    except Exception as e:
        logger.error(f"Error retrieving documents from Confluence: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def extract_review_info_batch(documents, batch_size=5):
    """
    Extract review information from multiple documents in batches
    """
    logger.info(f"Processing {len(documents)} documents for review information extraction")
    
    processed_documents = []
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    for batch_num in range(0, len(documents), batch_size):
        batch = documents[batch_num:batch_num + batch_size]
        current_batch_num = (batch_num // batch_size) + 1
        
        logger.info(f"Processing batch {current_batch_num}/{total_batches} ({len(batch)} documents)")
        
        # Process each document in the batch
        for doc in batch:
            try:
                processed_doc = extract_review_info_single(doc)
                if processed_doc:
                    processed_documents.append(processed_doc)
                else:
                    # If extraction failed, add document with default values
                    doc_with_defaults = doc.copy()
                    doc_with_defaults.update({
                        'has_review_block': False,
                        'updated_by': None,
                        'reviewed_by': None,
                        'updated_date': None,
                        'next_review_date': None
                    })
                    processed_documents.append(doc_with_defaults)
            except Exception as e:
                logger.error(f"Error processing document '{doc.get('title', 'Unknown')}': {str(e)}")
                # Add document with error state
                doc_with_error = doc.copy()
                doc_with_error.update({
                    'has_review_block': False,
                    'updated_by': None,
                    'reviewed_by': None,
                    'updated_date': None,
                    'next_review_date': None,
                    'processing_error': str(e)
                })
                processed_documents.append(doc_with_error)
    
    logger.info(f"Successfully processed {len(processed_documents)} documents")
    return processed_documents

def extract_review_info_single(document):
    """
    Extract review information from a single document using Claude, and extract the source URL and title if missing.
    """
    if not document or not document.get('content'):
        logger.warning(f"No content for document: {document.get('title', 'Unknown')}")
        return None
    try:
        content = document['content']
        # Skip documents with very little content
        if len(content.strip()) < 50:
            logger.info(f"Skipping document with minimal content: {document.get('title', 'Unknown')}")
            doc_with_defaults = document.copy()
            doc_with_defaults.update({
                'has_review_block': False,
                'updated_by': None,
                'reviewed_by': None,
                'updated_date': None,
                'next_review_date': None
            })
            return doc_with_defaults
        # Create prompt for Claude
        prompt = f"""
Analyze this document content and extract review information.

Look for review/tracking information such as:
- Review blocks or tables
- "Updated by", "Last updated by", "Author", "Modified by" fields
- "Reviewed by", "Reviewer", "Approved by" fields  
- "Updated date", "Last updated", "Modified date", "Created date" fields
- "Next review date", "Review due", "Next review", "Due date" fields
- Any metadata or tracking sections
- Always include source URL as **clickable links**
- Title of the document, if not present in metadata, use Heading or first line as title

Document Title: {document.get('title', 'Unknown Document')}
Document URL: {document.get('url', '')}
Data Source: {document.get('data_source', 'unknown')}

Content (first 3000 chars):
{content[:3000]}

Return ONLY a JSON object with exactly these fields:
{{
    "has_review_block": true/false,
    "updated_by": "name or null",
    "reviewed_by": "name or null", 
    "updated_date": "MM/DD/YYYY format or null",
    "next_review_date": "MM/DD/YYYY format or null",
    "source_url": "Source URL from Confluence or SharePoint document as a clickable link",
    "source_title": "Title of Confluence or SharePoint document."
}}

Be conservative - only set has_review_block to true if there's a clear review/tracking section.
"""
        response = bedrock_runtime.invoke_model(
            modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 5000,
                "temperature": 0,
                "system": "You are a document analysis assistant. Extract review/tracking information and return only valid JSON.",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
        )
        response_body = json.loads(response['body'].read())
        if 'content' in response_body and len(response_body['content']) > 0:
            content_text = response_body['content'][0]['text']
            # Extract JSON from response
            json_match = re.search(r'(\{[^}]*\})', content_text, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*?\})', content_text, re.DOTALL | re.MULTILINE)
            if json_match:
                json_str = json_match.group(1)
                review_info = json.loads(json_str)
                # Clean and validate
                cleaned_review_info = {
                    'has_review_block': bool(review_info.get('has_review_block', False)),
                    'updated_by': review_info.get('updated_by') if review_info.get('updated_by') not in ['null', None] else None,
                    'reviewed_by': review_info.get('reviewed_by') if review_info.get('reviewed_by') not in ['null', None] else None,
                    'updated_date': review_info.get('updated_date') if review_info.get('updated_date') not in ['null', None] else None,
                    'next_review_date': review_info.get('next_review_date') if review_info.get('next_review_date') not in ['null', None] else None
                }
                # If Claude provides a source_url or source_title, use them if missing
                doc_with_review = document.copy()
                doc_with_review.update(cleaned_review_info)
                if (not doc_with_review.get('url') or doc_with_review['url'] in [None, '', 'No URL Available']) and review_info.get('source_url'):
                    doc_with_review['url'] = review_info['source_url']
                if (not doc_with_review.get('title') or doc_with_review['title'] in [None, '', 'Unknown Document']) and review_info.get('source_title'):
                    doc_with_review['title'] = review_info['source_title']
                return doc_with_review
        # If extraction failed, return with default values
        doc_with_defaults = document.copy()
        doc_with_defaults.update({
            'has_review_block': False,
            'updated_by': None,
            'reviewed_by': None,
            'updated_date': None,
            'next_review_date': None
        })
        return doc_with_defaults
    except Exception as e:
        logger.error(f"Error extracting review info for document '{document.get('title', 'Unknown')}': {str(e)}")
        return None

def get_next_assignee():
    """Get next engineer using round robin algorithm"""
    try:
        # Get all engineers (exclude control record)
        response = user_manager_table.scan()
        engineers = [item['PK'] for item in response['Items'] if item['PK'] != 'last_assigned_index']
        
        if not engineers:
            logger.error("No engineers found in user_manager_map table")
            return None, None
        
        # Get current index
        control_item = user_manager_table.get_item(Key={'PK': 'last_assigned_index'})
        current_index = control_item.get('Item', {}).get('index', -1)
        current_index = int(current_index)  # Ensure integer type
        
        # Calculate next assignment
        next_index = (current_index + 1) % len(engineers)
        next_index = int(next_index)  # Ensure integer type
        assigned_engineer = engineers[next_index]
        
        # Get manager for assigned engineer
        manager_response = user_manager_table.get_item(Key={'PK': assigned_engineer})
        manager_upn = manager_response.get('Item', {}).get('ManagerUserPrincipalName', 'Unknown')
        
        # Update the index
        user_manager_table.put_item(Item={
            'PK': 'last_assigned_index',
            'index': next_index
        })
        
        logger.info(f"Assigned to: {assigned_engineer}, Manager: {manager_upn}")
        return assigned_engineer, manager_upn
    
    except Exception as e:
        logger.error(f"Error in round robin assignment: {str(e)}")
        return None, None

def determine_document_status_with_assignment(doc_with_review_info):
    """Enhanced status determination with assignment logic"""
    try:
        today = datetime.datetime.now()
        review_threshold_days = int(get_parameter('/document-review/REVIEW_THRESHOLD_DAYS', '60'))
        threshold_date = today + datetime.timedelta(days=review_threshold_days)
        
        status = 'compliant'
        issue = None
        needs_assignment = False
        
        # Determine status
        if not doc_with_review_info.get('has_review_block', False):
            status = 'missing_block'
            issue = 'Missing review block'
            needs_assignment = True
        elif not doc_with_review_info.get('next_review_date'):
            status = 'missing_date'
            issue = 'Missing next review date'
            needs_assignment = True
        else:
            try:
                next_review_date = None
                date_formats = ['%m/%d/%Y', '%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y', '%m-%d-%Y']
                
                for date_format in date_formats:
                    try:
                        next_review_date = datetime.datetime.strptime(doc_with_review_info['next_review_date'], date_format)
                        break
                    except ValueError:
                        continue
                
                if next_review_date:
                    days_until_review = (next_review_date - today).days
                    if next_review_date < today:
                        status = 'expired'
                        issue = f'Review date expired ({abs(days_until_review)} days overdue)'
                        needs_assignment = True
                    elif next_review_date <= threshold_date:
                        status = 'due_soon'
                        issue = f'Review due within {review_threshold_days} days ({days_until_review} days remaining)'
                        needs_assignment = True
                    else:
                        status = 'compliant'
                        issue = None
                        needs_assignment = False
                else:
                    status = 'invalid_date'
                    issue = 'Invalid review date format'
                    needs_assignment = True
            except Exception as e:
                status = 'invalid_date'
                issue = f'Error parsing review date: {str(e)}'
                needs_assignment = True
        
        # Assign if needed
        assigned_to = None
        manager_upn = None
        if needs_assignment:
            assigned_to, manager_upn = get_next_assignee()
        
        return {
            'status': status,
            'issue': issue,
            'needs_review': needs_assignment,
            'assigned_to': assigned_to,
            'manager': manager_upn,
            'document_info': doc_with_review_info
        }
    
    except Exception as e:
        logger.error(f"Error determining document status: {str(e)}")
        return {
            'status': 'error',
            'issue': f'Processing error: {str(e)}',
            'needs_review': False,
            'assigned_to': None,
            'manager': None,
            'document_info': doc_with_review_info
        }

def store_document_status(doc_result):
    """Store document review status in DynamoDB"""
    try:
        doc_info = doc_result['document_info']
        timestamp = datetime.datetime.utcnow().isoformat() + 'Z'
        
        # Generate document ID if not available
        doc_id = doc_info.get('document_id') or str(uuid.uuid4())
        
        # Check if doc_review_table uses 'PK' or 'doc_id' as primary key
        # Try to determine table schema first
        try:
            # Test with a sample query to understand table structure
            sample_response = doc_review_table.get_item(Key={'doc_id': 'test'})
            primary_key_field = 'doc_id'
        except Exception:
            try:
                sample_response = doc_review_table.get_item(Key={'PK': 'test'})
                primary_key_field = 'PK'
            except Exception:
                # Default to PK if both fail
                primary_key_field = 'PK'
                logger.warning("Could not determine primary key field, defaulting to 'PK'")
        
        item = {
            primary_key_field: doc_id,
            'title': doc_info.get('title'),
            'url': doc_info.get('source'),
            'status': doc_result['status'],
            'issue': doc_result.get('issue'),
            'has_review_block': doc_info.get('has_review_block', False),
            'updated_by': doc_info.get('updated_by'),
            'reviewed_by': doc_info.get('reviewed_by'),
            'updated_date': doc_info.get('updated_date'),
            'next_review_date': doc_info.get('next_review_date'),
            'data_source': doc_info.get('data_source', 'unknown'),
            'author': doc_info.get('author'),
            'last_modified': doc_info.get('last_modified'),
            'created_ts': timestamp,
            'last_updated_ts': timestamp,
            'notifications_sent': 0,
            'escalated': False,
            'resolution_ts': None,
            'needs_review': doc_result['needs_review']
        }
        # Only fallback if both title and url are missing or empty
        if not item.get('title'):
            item['title'] = 'Untitled Document'
        if not item.get('url'):
            item['url'] = 'No URL Available'
        # Add assignment info if document needs review
        if doc_result['assigned_to'] and doc_result['manager']:
            item['assigned_to_upn'] = doc_result['assigned_to']
            item['manager_upn'] = doc_result['manager']
        
        # Convert any float values to Decimal for DynamoDB
        for key, value in item.items():
            if isinstance(value, float):
                item[key] = Decimal(str(value))
        
        # Remove None values to avoid DynamoDB issues
        item = {k: v for k, v in item.items() if v is not None}
        
        doc_review_table.put_item(Item=item)
        logger.info(f"Stored document: {doc_id} | Status: {doc_result['status']} | Key field: {primary_key_field}")
        
        return doc_id
    
    except Exception as e:
        logger.error(f"Error storing document status: {str(e)}")
        logger.error(f"Document info: {doc_info.get('title', 'Unknown')} | {doc_info.get('url', '')}")
        return None

def lambda_handler(event, context):
    """
    Lambda handler to process ALL documents from Confluence
    """
    try:
        start_time = datetime.datetime.now()
        logger.info(f"Starting bulk document review analysis at {start_time}")
        # Step 1: Retrieve ALL documents from Confluence
        logger.info("Step 1: Retrieving all documents from Confluence...")
        all_documents = retrieve_all_confluence_documents()
        if not all_documents:
            logger.warning("No documents found in Confluence space")
            return {
                'statusCode': 404,
                'body': json.dumps({
                    'message': 'No documents found in Confluence space',
                    'space_key': 'CLOUD'
                })
            }
        logger.info(f"Retrieved {len(all_documents)} documents from Confluence")
        # Step 2: Extract review information from all documents
        logger.info("Step 2: Extracting review information from all documents...")
        docs_with_review_info = extract_review_info_batch(all_documents)
        # Step 3: Store review info in DynamoDB
        logger.info("Step 3: Storing review information in DynamoDB...")
        for doc in docs_with_review_info:
            try:
                # Determine status and assignment
                doc_result = determine_document_status_with_assignment(doc)
                store_document_status(doc_result)
            except Exception as e:
                logger.error(f"Error storing review info for document {doc.get('title', 'Unknown')}: {str(e)}")
                continue
        # Step 4: Build metadata-only list for response
        metadata_list = []
        for doc in docs_with_review_info:
            metadata_list.append({
                'doc_id': doc.get('document_id'),
                'title': doc.get('title'),
                'url': doc.get('url'),
                'data_source': doc.get('data_source'),
                'author': doc.get('author'),
                'last_modified': doc.get('last_modified')
            })
        return {
            'statusCode': 200,
            'body': json.dumps(metadata_list, default=str, indent=2)
        }
    except Exception as e:
        logger.error(f"Error in bulk document review analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'An unexpected error occurred during bulk document analysis'
            })
        }