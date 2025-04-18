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
    """
    Lambda function to upload images to S3 bucket or create folders
    
    Event format:
    {
        "image_key": "filename.jpg",  # Required: Name of the file to be saved or folder path (ending with /)
        "image_data": "base64string",  # Required for files: Base64 encoded image data; For folders: can be empty string
        "prefix": "folder/"           # Optional: Folder path to store the image in
    }
    """
    # CORS headers to include in all responses
    cors_headers = {
        'Access-Control-Allow-Origin': '*',  # Or specify your domain: 'http://localhost:3000'
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'OPTIONS,POST,PUT'
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
        
        # Check if it's a folder creation request (key ends with '/')
        is_folder = image_key.endswith('/')
        
        # Validate parameters
        if not image_key:
            logger.error("Missing required parameter: image_key")
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({
                    'message': 'Missing required parameter: image_key'
                })
            }
        
        # For non-folder uploads, image_data is required
        if not is_folder and not image_data:
            logger.error("Missing required parameter: image_data for file upload")
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({
                    'message': 'Missing required parameter: image_data for file upload'
                })
            }
        
        # Get the folder prefix from the request
        folder_prefix = body.get('prefix', '')
        
        # Set up S3 prefix
        user_prefix = 'users/'
        prefix = user_prefix + folder_prefix if folder_prefix else user_prefix
        
        # Construct the full S3 key
        s3_key = f"{prefix}{image_key}"
        
        logger.info(f"Processing request for key: {s3_key}, is_folder: {is_folder}")
        
        # For file uploads, decode base64 image data
        if not is_folder:
            try:
                image_binary = base64.b64decode(image_data)
            except Exception as e:
                logger.error(f"Base64 decode error: {str(e)}")
                return {
                    'statusCode': 400,
                    'headers': cors_headers,
                    'body': json.dumps({
                        'message': 'Invalid base64 image data'
                    })
                }
        else:
            # For folders, use empty bytes
            image_binary = b''
        
        # Upload to S3
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
        
        try:
            # Set content type based on whether this is a folder or file
            if is_folder:
                # Use 'application/x-directory' for folders
                content_type = 'application/x-directory'
            else:
                # Get file extension to set correct content type
                file_ext = image_key.lower().split('.')[-1] if '.' in image_key else 'jpg'
                content_type = f'image/{file_ext}' if file_ext != 'jpg' else 'image/jpeg'
            
            # Log the upload parameters
            logger.info(f"Uploading to S3: Bucket={bucket_name}, Key={s3_key}, ContentType={content_type}")
            
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
                'headers': cors_headers,
                'body': json.dumps({
                    'message': f'Failed to upload image: {error_code} - {error_message}'
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
        
        # Return success response with type-specific message
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps({
                'message': 'Folder created successfully' if is_folder else 'Image uploaded successfully',
                'key': s3_key,
                'bucket': bucket_name,
                'type': 'folder' if is_folder else 'file'
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