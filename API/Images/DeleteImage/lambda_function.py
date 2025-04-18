import json
import os
import boto3
from botocore.exceptions import ClientError
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    Lambda function to delete images or folders from S3 bucket
    
    Event format:
    {
        "image_key": "path/to/image.jpg" or "path/to/folder/"  # Path to image file or folder (folders end with /)
    }
    
    For folders, this function deletes all objects with the specified prefix.
    """
    # CORS headers to include in all responses
    cors_headers = {
        'Access-Control-Allow-Origin': '*',  # Or specify your domain: 'http://localhost:3000'
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'OPTIONS,POST,DELETE'
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
        # Log the incoming event
        logger.info(f"Received event: {json.dumps(event)}")
        
        # Get request body - handle both direct event data and body field
        if isinstance(event, str):
            body = json.loads(event)
        elif isinstance(event, dict):
            if 'body' in event:
                body = event['body'] if isinstance(event['body'], dict) else json.loads(event['body'])
            else:
                body = event
        else:
            body = {}
            
        logger.info(f"Processed request body: {json.dumps(body)}")
        
        # Get image key
        image_key = body.get('image_key', '')
        
        if not image_key:
            logger.error("Missing required parameter")
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({
                    'message': 'Missing required parameter: image_key'
                })
            }
        
        # Determine if this is a folder delete (key ends with /)
        is_folder = image_key.endswith('/')
        logger.info(f"Request type: {'folder' if is_folder else 'file'} deletion")
        
        # Set up S3 client
        s3_client = boto3.client('s3')
        bucket_name = os.environ.get('BUCKET_NAME')
        
        if not bucket_name:
            logger.error("BUCKET_NAME environment variable not set")
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({
                    'message': 'Server configuration error: BUCKET_NAME not set'
                })
            }
        
        deleted_count = 0
        
        try:
            if is_folder:
                # For folder delete, delete all objects with this prefix
                logger.info(f"Folder deletion: listing objects with prefix: {image_key}")
                
                # List all objects in the folder
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(
                    Bucket=bucket_name,
                    Prefix=image_key
                )
                
                # Check if the folder exists (has any contents)
                objects_found = False
                objects_to_delete = []
                
                for page in pages:
                    if 'Contents' in page:
                        objects_found = True
                        for obj in page['Contents']:
                            objects_to_delete.append({'Key': obj['Key']})
                
                if not objects_found:
                    logger.warning(f"Folder doesn't exist or is empty: {image_key}")
                    return {
                        'statusCode': 404,
                        'headers': cors_headers,
                        'body': json.dumps({
                            'message': f'Folder to delete doesn\'t exist or is empty: {image_key}'
                        })
                    }
                
                # Delete all objects in batch
                if objects_to_delete:
                    logger.info(f"Deleting {len(objects_to_delete)} objects from folder {image_key}")
                    response = s3_client.delete_objects(
                        Bucket=bucket_name,
                        Delete={
                            'Objects': objects_to_delete,
                            'Quiet': False
                        }
                    )
                    
                    # Log the results
                    deleted_count = len(response.get('Deleted', []))
                    errors = response.get('Errors', [])
                    if errors:
                        error_summary = [f"{e['Key']}: {e['Code']}" for e in errors]
                        logger.error(f"Errors during bulk deletion: {error_summary}")
                    
                    logger.info(f"Successfully deleted {deleted_count} objects, errors: {len(errors)}")
            else:
                # Single file delete - verify object exists
                try:
                    s3_client.head_object(
                        Bucket=bucket_name,
                        Key=image_key
                    )
                except ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        logger.error(f"Object to delete does not exist: {image_key}")
                        return {
                            'statusCode': 404,
                            'headers': cors_headers,
                            'body': json.dumps({
                                'message': f'Image to delete does not exist: {image_key}'
                            })
                        }
                    else:
                        raise e

                # Delete single object
                logger.info(f"Deleting single file: {image_key}")
                response = s3_client.delete_object(
                    Bucket=bucket_name,
                    Key=image_key
                )
                logger.info(f"S3 delete response: {response}")
                deleted_count = 1
                
                # Verify deletion success
                try:
                    s3_client.head_object(
                        Bucket=bucket_name,
                        Key=image_key
                    )
                    logger.error("Delete failed: object still exists")
                    raise Exception("Delete operation failed to remove object")
                except ClientError as e:
                    if e.response['Error']['Code'] == '404':
                        logger.info("Delete verification successful: object has been removed")
                    else:
                        raise e
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            logger.error(f"S3 delete error: Code={error_code}, Message={error_message}")
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({
                    'message': f'Failed to delete {"folder" if is_folder else "image"}: {error_code} - {error_message}'
                })
            }
        
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps({
                'message': 'Folder deleted successfully' if is_folder else 'Image deleted successfully',
                'key': image_key,
                'count': deleted_count,
                'bucket': bucket_name
            })
        }
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': cors_headers,
            'body': json.dumps({
                'message': f'Internal server error: {str(e)}'
            })
        } 