import json
import boto3
import os
import logging
import datetime
import traceback
import re

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name="us-east-1")
bedrock_runtime = boto3.client('bedrock-runtime', region_name="us-east-1")

def retrieve_specific_page(url="https://cbre-jira.atlassian.net/wiki/spaces/CLOUD/pages/583696664/Cloud+Vendor+Contacts"):
# def retrieve_specific_page(url="https://cbre-jira.atlassian.net/wiki/spaces/CLOUD/pages/39590831/AWS+Release+Notes"):
    try:
        knowledge_base_id = "01ANQMZB9P"
        page_title = "Cloud Vendor Contacts"
        # page_title = "AWS Release Notes"
        
        # Try multiple search strategies
        search_queries = [
            f"\"{page_title}\"",
            "Cloud Vendor Contacts", "AWS Release Notes", "vendor contacts","CLOUD space vendor contacts","583696664","cloud vendor"]
        
        all_documents = []
        
        for query in search_queries:
            try:
                logger.info(f"Trying search query: {query}")
                
                retrieve_params = {
                    'knowledgeBaseId': knowledge_base_id,
                    'retrievalQuery': {
                        'text': query
                    },
                    'retrievalConfiguration': {
                        'vectorSearchConfiguration': {
                            'numberOfResults': 20  # Get more results
                        }
                    }
                }
                
                response = bedrock_agent_runtime.retrieve(**retrieve_params)
                
                # Process results
                for result in response.get('retrievalResults', []):
                    content_text = ""
                    if 'content' in result and 'text' in result['content']:
                        content_text = result['content']['text']
                    
                    doc = {
                        'document_id': '',
                        'title': 'Unknown Document',
                        'url': '',
                        'content': content_text,
                        'data_source': 'confluence',
                        'score': result.get('score', 0)  # Add relevance score
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
                                if '/' in doc['url']:
                                    doc['document_id'] = doc['url'].split('/')[-1]
                                else:
                                    doc['document_id'] = doc['url']
                    
                    # Only add if we haven't seen this document before
                    if not any(existing_doc['url'] == doc['url'] for existing_doc in all_documents):
                        all_documents.append(doc)
                
                logger.info(f"Query '{query}' returned {len(response.get('retrievalResults', []))} results")
                
            except Exception as e:
                logger.warning(f"Search query '{query}' failed: {str(e)}")
                continue
        
        logger.info(f"Total unique documents found: {len(all_documents)}")
        
        # Log all found documents for debugging
        for i, doc in enumerate(all_documents):
            logger.info(f"Document {i+1}: Title='{doc['title']}', URL='{doc['url']}', Score={doc.get('score', 'N/A')}")
        
        # Try to find the target document using multiple matching strategies
        target_document = None
        
        # Strategy 1: Exact URL match
        for doc in all_documents:
            if url in doc.get('url', ''):
                logger.info(f"Found document by exact URL match: {doc['title']}")
                target_document = doc
                break
        
        # Strategy 2: Page ID match
        if not target_document:
            page_id = "583696664"
            for doc in all_documents:
                if page_id in doc.get('url', ''):
                    logger.info(f"Found document by page ID match: {doc['title']}")
                    target_document = doc
                    break
        
        # Strategy 3: Title similarity match
        if not target_document:
            target_title_lower = page_title.lower()
            for doc in all_documents:
                doc_title_lower = doc.get('title', '').lower()
                # Check for exact match or high similarity
                if (target_title_lower == doc_title_lower or 
                    target_title_lower in doc_title_lower or
                    all(word in doc_title_lower for word in target_title_lower.split())):
                    logger.info(f"Found document by title match: {doc['title']}")
                    target_document = doc
                    break
        
        # Strategy 4: Content-based match
        if not target_document:
            keywords = ["vendor", "contact", "cloud"]
            for doc in all_documents:
                content_lower = doc.get('content', '').lower()
                title_lower = doc.get('title', '').lower()
                
                # Check if document contains relevant keywords
                keyword_matches = sum(1 for keyword in keywords if keyword in content_lower or keyword in title_lower)
                if keyword_matches >= 2:  # At least 2 keywords match
                    logger.info(f"Found document by keyword match: {doc['title']} (matches: {keyword_matches})")
                    target_document = doc
                    break
        
        # Strategy 5: Take the highest scoring document if still no match
        if not target_document and all_documents:
            # Sort by score descending
            sorted_docs = sorted(all_documents, key=lambda x: x.get('score', 0), reverse=True)
            target_document = sorted_docs[0]
            logger.info(f"Using highest scoring document: {target_document['title']} (score: {target_document.get('score', 'N/A')})")
        
        if not target_document:
            logger.error(f"Could not find document with URL: {url}")
            logger.error("Available documents:")
            for doc in all_documents:
                logger.error(f"  - {doc['title']} | {doc['url']}")
            return None
        
        logger.info(f"Selected document: {target_document['title']} | {target_document['url']}")
        return target_document
    
    except Exception as e:
        logger.error(f"Error retrieving specific page: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def extract_review_info(document):
    """
    Use Claude to extract review information with improved prompting
    """
    if not document or not document.get('content'):
        logger.warning("No document content to extract review info from")
        return None
    
    try:
        content = document['content']
        logger.info(f"Document content length: {len(content)} characters")
        
        # Create a more specific prompt for Claude
        prompt = f"""
Analyze this Confluence page content and extract review information.

Look for review/tracking information that typically appears in Confluence pages, which may include:
- Review blocks or tables
- "Updated by" or "Last updated by" fields
- "Reviewed by" fields
- "Updated date" or "Last updated" dates
- "Next review date" or "Review due" dates
- Metadata or tracking information

Document Title: {document.get('title', 'Unknown Document')}
Document URL: {document.get('url', '')}

Content:
{content}

Return ONLY a JSON object with exactly these fields:
{{
    "has_review_block": true/false,
    "updated_by": "name or null",
    "reviewed_by": "name or null", 
    "updated_date": "MM/DD/YYYY format or null",
    "next_review_date": "MM/DD/YYYY format or null"
}}

If you cannot find specific review information, set the corresponding fields to null but indicate whether there appears to be any form of review/tracking section.
"""
        
        logger.info("Invoking Claude to extract review information")
        response = bedrock_runtime.invoke_model(
            modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "temperature": 0,
                "system": "You are a document analysis assistant. Extract review/tracking information from Confluence pages and return only valid JSON.",
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
            logger.info(f"Claude response: {content_text}")
            
            # Extract JSON from the response
            try:
                # Try to find JSON in the response
                json_match = re.search(r'(\{[^}]+\})', content_text, re.DOTALL)
                if not json_match:
                    # Try to find JSON spanning multiple lines
                    json_match = re.search(r'(\{.*?\})', content_text, re.DOTALL | re.MULTILINE)
                
                if json_match:
                    json_str = json_match.group(1)
                    logger.info(f"Extracted JSON string: {json_str}")
                    
                    review_info = json.loads(json_str)
                    
                    # Validate and clean the review info
                    cleaned_review_info = {
                        'has_review_block': bool(review_info.get('has_review_block', False)),
                        'updated_by': review_info.get('updated_by') if review_info.get('updated_by') != 'null' else None,
                        'reviewed_by': review_info.get('reviewed_by') if review_info.get('reviewed_by') != 'null' else None,
                        'updated_date': review_info.get('updated_date') if review_info.get('updated_date') != 'null' else None,
                        'next_review_date': review_info.get('next_review_date') if review_info.get('next_review_date') != 'null' else None
                    }
                    
                    # Add the review info to the document
                    doc_with_review = document.copy()
                    doc_with_review.update(cleaned_review_info)
                    
                    logger.info(f"Successfully extracted review info: {cleaned_review_info}")
                    return doc_with_review
                else:
                    logger.warning("Could not find JSON in Claude's response")
                    logger.warning(f"Full response: {content_text}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from model response: {e}")
                logger.warning(f"Raw response: {content_text}")
        
        # If extraction failed, return document with default values
        doc_with_review = document.copy()
        doc_with_review.update({
            'has_review_block': False,
            'updated_by': None,
            'reviewed_by': None,
            'updated_date': None,
            'next_review_date': None
        })
        
        return doc_with_review
    
    except Exception as e:
        logger.error(f"Error extracting review info: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def determine_document_status(doc_with_review_info):
    """
    Determine if the document needs review based on review dates
    """
    if not doc_with_review_info:
        return {
            'status': 'Error',
            'issue': 'Failed to process document'
        }
    
    today = datetime.datetime.now()
    review_threshold_days = 60
    threshold_date = today + datetime.timedelta(days=review_threshold_days)
    
    issue = None
    
    # Check if the document has a review block
    if not doc_with_review_info.get('has_review_block', False):
        issue = "Missing review block"
    
    # Check if the document has a next review date
    elif doc_with_review_info.get('next_review_date'):
        try:
            # Try to parse the date - handle different formats
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
                    issue = f"Review date expired ({abs(days_until_review)} days overdue)"
                elif next_review_date <= threshold_date:
                    issue = f"Review due within 60 days ({days_until_review} days remaining)"
                else:
                    issue = None  # Document is up to date
            else:
                issue = "Invalid review date format"
        except Exception as e:
            logger.warning(f"Error parsing review date: {e}")
            issue = "Invalid review date format"
    else:
        issue = "Missing next review date"
    
    # Create the status report
    status_report = doc_with_review_info.copy()
    status_report['issue'] = issue
    status_report['needs_review'] = issue is not None
    status_report['analysis_date'] = today.strftime('%Y-%m-%d %H:%M:%S')
    
    return status_report

def lambda_handler(event, context):
    """
    Enhanced Lambda handler with better error handling and logging
    """
    try:
        logger.info(f"Starting document review test at {datetime.datetime.now()}")
        
        # Step 1: Retrieve the specific page
        logger.info("Step 1: Retrieving document from knowledge base...")
        document = retrieve_specific_page()
        
        if not document:
            logger.error("Failed to retrieve document from knowledge base")
            return {
                'statusCode': 404,
                'body': json.dumps({
                    'message': 'Could not find the specified Confluence page',
                    'url': "https://cbre-jira.atlassian.net/wiki/spaces/CLOUD/pages/583696664/Cloud+Vendor+Contacts",
                    'suggestion': 'Check if the document exists in the knowledge base and verify the knowledge base ID'
                })
            }
        
        logger.info(f"Successfully retrieved document: {document['title']}")
        
        # Step 2: Extract review information
        logger.info("Step 2: Extracting review information...")
        doc_with_review_info = extract_review_info(document)
        
        if not doc_with_review_info:
            logger.error("Failed to extract review information")
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'message': 'Failed to extract review information',
                    'document': {
                        'title': document['title'],
                        'url': document['url']
                    }
                })
            }
        
        # Step 3: Determine document status
        logger.info("Step 3: Determining document status...")
        status_report = determine_document_status(doc_with_review_info)
        
        logger.info(f"Analysis complete. Document needs review: {status_report.get('needs_review', 'unknown')}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Successfully analyzed document',
                'document_status': status_report
            }, default=str, indent=2)
        }
    
    except Exception as e:
        logger.error(f"Error in document review test: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'message': 'An unexpected error occurred during document analysis'
            })
        }