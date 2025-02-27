import json
import os
import boto3
from botocore.exceptions import ClientError
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
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
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST,DELETE'
                },
                'body': json.dumps({
                    'message': 'Missing required parameter: image_key'
                })
            }
        
        # Set up S3 client
        s3_client = boto3.client('s3')
        bucket_name = os.environ.get('BUCKET_NAME')
        
        if not bucket_name:
            logger.error("BUCKET_NAME environment variable not set")
            return {
                'statusCode': 500,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST,DELETE'
                },
                'body': json.dumps({
                    'message': 'Server configuration error: BUCKET_NAME not set'
                })
            }
            
        logger.info(f"Deleting from bucket: {bucket_name}, key: {image_key}")
        
        try:
            # Verify object exists
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
                        'headers': {
                            'Access-Control-Allow-Origin': '*',
                            'Access-Control-Allow-Headers': 'Content-Type',
                            'Access-Control-Allow-Methods': 'OPTIONS,POST,DELETE'
                        },
                        'body': json.dumps({
                            'message': f'Image to delete does not exist: {image_key}'
                        })
                    }
                else:
                    raise e

            # Delete object
            response = s3_client.delete_object(
                Bucket=bucket_name,
                Key=image_key
            )
            logger.info(f"S3 delete response: {response}")
            
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
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST,DELETE'
                },
                'body': json.dumps({
                    'message': f'Failed to delete image: {error_code} - {error_message}'
                })
            }
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,DELETE'
            },
            'body': json.dumps({
                'message': 'Image deleted successfully',
                'key': image_key,
                'bucket': bucket_name
            })
        }
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,DELETE'
            },
            'body': json.dumps({
                'message': f'Internal server error: {str(e)}'
            })
        } 