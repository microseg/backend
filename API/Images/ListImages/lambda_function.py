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
    try:
        # Get bucket name from environment variables
        bucket_name = os.environ['BUCKET_NAME']
        if not bucket_name:
            logger.error("BUCKET_NAME environment variable not set")
            return {
                'statusCode': 500,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': 'Server configuration error: BUCKET_NAME not set'
                })
            }
        
        # Get parameters and add users/ prefix
        user_prefix = 'users/'
        folder_prefix = event.get('prefix', '')
        prefix = user_prefix + folder_prefix if folder_prefix else user_prefix
            
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Build ListObjectsV2 parameters
        list_params = {
            'Bucket': bucket_name,
            'Prefix': prefix
        }
        
        try:
            # Get object list
            response = s3_client.list_objects_v2(**list_params)
            
            # If bucket is empty
            if 'Contents' not in response:
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'images': []
                    })
                }
                
            # Extract image information
            images = [{
                'key': obj['Key'],
                'size': obj['Size'],
                'last_modified': obj['LastModified'].isoformat(),
                'etag': obj['ETag']
            } for obj in response['Contents']]
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'images': images,
                    'count': len(images)
                })
            }
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            logger.error(f"S3 list error: Code={error_code}, Message={error_message}")
            return {
                'statusCode': 500,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'error': f'Failed to list images: {error_message}'
                })
            }
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f'Internal server error: {str(e)}'
            })
        } 