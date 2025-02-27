import json
import os
import base64
import boto3
from botocore.exceptions import ClientError
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    try:
        # Get the request body - handle both direct event data and body field
        if isinstance(event, str):
            body = json.loads(event)
        elif isinstance(event, dict):
            if 'body' in event:
                body = event['body'] if isinstance(event['body'], dict) else json.loads(event['body'])
            else:
                body = event
        else:
            body = {}
            
        # Get the image key and base64 data from the request
        image_key = body.get('image_key', '')
        image_data = body.get('image_data', '')
        
        if not image_key or not image_data:
            logger.error("Missing required parameters")
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST'
                },
                'body': json.dumps({
                    'message': 'Missing required parameters: image_key or image_data'
                })
            }
        
        # Get the folder prefix from the request
        folder_prefix = body.get('prefix', '')
        
        # Set up S3 prefix
        user_prefix = 'users/'
        prefix = user_prefix + folder_prefix if folder_prefix else user_prefix
        
        # Construct the full S3 key
        s3_key = f"{prefix}{image_key}"
        
        # Decode base64 image data
        try:
            image_binary = base64.b64decode(image_data)
        except Exception as e:
            logger.error(f"Base64 decode error: {str(e)}")
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST'
                },
                'body': json.dumps({
                    'message': 'Invalid base64 image data'
                })
            }
        
        # Upload to S3
        s3_client = boto3.client('s3')
        bucket_name = os.environ.get('BUCKET_NAME')
        
        if not bucket_name:
            logger.error("BUCKET_NAME environment variable not set")
            return {
                'statusCode': 500,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST'
                },
                'body': json.dumps({
                    'message': 'Server configuration error: BUCKET_NAME not set'
                })
            }
        
        try:
            # Get file extension to set correct content type
            file_ext = image_key.lower().split('.')[-1] if '.' in image_key else 'jpg'
            content_type = f'image/{file_ext}' if file_ext != 'jpg' else 'image/jpeg'
            
            response = s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=image_binary,
                ContentType=content_type
            )
            
            # Verify the upload
            try:
                s3_client.head_object(
                    Bucket=bucket_name,
                    Key=s3_key
                )
            except ClientError as e:
                logger.error(f"Upload verification failed: {str(e)}")
                raise Exception("Upload succeeded but verification failed")
                
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            logger.error(f"S3 upload error: Code={error_code}, Message={error_message}")
            logger.error(f"Full error response: {e.response}")
            return {
                'statusCode': 500,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST'
                },
                'body': json.dumps({
                    'message': f'Failed to upload image: {error_code} - {error_message}'
                })
            }
        
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {
                'statusCode': 500,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Headers': 'Content-Type',
                    'Access-Control-Allow-Methods': 'OPTIONS,POST'
                },
                'body': json.dumps({
                    'message': f'Internal server error: {str(e)}'
                })
            }
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({
                'message': 'Image uploaded successfully',
                'key': s3_key,
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
                'Access-Control-Allow-Methods': 'OPTIONS,POST'
            },
            'body': json.dumps({
                'message': f'Internal server error: {str(e)}'
            })
        } 