import json
import boto3
import os
from botocore.exceptions import ClientError
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    Lambda function to list all images in S3 bucket
    
    Event format:
    {
        "prefix": "folder/"  # Optional parameter to list images in a specific folder
    }
    """
    # CORS headers to include in all responses
    cors_headers = {
        'Access-Control-Allow-Origin': '*',  # Or specify your domain: 'http://localhost:3000'
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
    }
    
    # Handle OPTIONS requests (preflight CORS requests)
    if event.get('httpMethod') == 'OPTIONS':
        logger.info("Handling OPTIONS preflight request")
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps({})
        }
        
    try:
        # Get bucket name from environment variables
        bucket_name = os.environ['BUCKET_NAME']
        if not bucket_name:
            logger.error("BUCKET_NAME environment variable not set")
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({
                    'error': 'Server configuration error: BUCKET_NAME not set'
                })
            }
        
        # Get user prefix from request
        user_prefix = 'users/'
        folder_prefix = event.get('prefix', '')
        
        # Log the received prefix for debugging
        logger.info(f"Received prefix: {folder_prefix}")
        
        # Construct the full S3 prefix
        if folder_prefix:
            # Handle different prefix formats
            if folder_prefix.endswith('/'):
                # Prefix already has trailing slash
                prefix = f"{user_prefix}{folder_prefix}"
            elif '/' in folder_prefix:
                # User is navigating into a subfolder, ensure trailing slash
                prefix = f"{user_prefix}{folder_prefix}/"
            else:
                # User ID only, add trailing slash
                prefix = f"{user_prefix}{folder_prefix}/"
        else:
            # No prefix, just list user folders
            prefix = user_prefix
            
        # Log the constructed S3 prefix for debugging
        logger.info(f"Constructed S3 prefix: {prefix}")
            
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Build ListObjectsV2 parameters
        list_params = {
            'Bucket': bucket_name,
            'Prefix': prefix,
            'Delimiter': '/'  # Use delimiter to group by folders
        }
        
        try:
            # Get object list
            response = s3_client.list_objects_v2(**list_params)
            
            # Log response structure for debugging
            logger.info(f"S3 response: {str(response)[:200]}...")
            
            # Initialize empty lists for files and folders
            files = []
            folders = []
            
            # Get files from Contents
            if 'Contents' in response:
                # Filter out the current folder marker itself (key ends with /)
                for obj in response['Contents']:
                    if not obj['Key'].endswith('/'):
                        # For files, we keep the full path in 'key' but simplify the name for display
                        file_name = obj['Key'].split('/')[-1]
                        
                        files.append({
                            'key': obj['Key'],
                            'name': file_name,
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'].isoformat(),
                            'etag': obj['ETag'],
                            'type': 'file'
                        })
            
            # Get folders from CommonPrefixes
            if 'CommonPrefixes' in response:
                for common_prefix in response['CommonPrefixes']:
                    prefix_key = common_prefix['Prefix']
                    
                    # Extract just the folder name by removing the path prefix and trailing slash
                    folder_parts = prefix_key.rstrip('/').split('/')
                    folder_name = folder_parts[-1]
                    
                    folders.append({
                        'key': prefix_key,
                        'name': folder_name,
                        'type': 'folder'
                    })
            
            # For return, we send separate lists and don't combine them
            # This lets frontend decide how to display them
            return {
                'statusCode': 200,
                'headers': cors_headers,
                'body': json.dumps({
                    'images': files,
                    'folders': folders,
                    'count': len(files) + len(folders)
                })
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            logger.error(f"S3 list error: Code={error_code}, Message={error_message}")
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({
                    'error': f'Failed to list images: {error_message}'
                })
            }
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': cors_headers,
            'body': json.dumps({
                'error': f'Internal server error: {str(e)}'
            })
        } 