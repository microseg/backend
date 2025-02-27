import json
import boto3
import os
from botocore.exceptions import ClientError

def get_stage_from_context(context):
    """
    Get stage information from Lambda context
    Extract stage name from function alias or version
    """
    alias_or_version = context.function_version
    if alias_or_version == '$LATEST':
        return 'dev'  # Default to dev
    return alias_or_version  # Return alias as stage name

def get_parameter(parameter_name, context):
    """
    Get parameter value from Parameter Store
    """
    stage = get_stage_from_context(context)
    full_parameter_name = f'{parameter_name}-{stage}'
    
    ssm = boto3.client('ssm')
    try:
        response = ssm.get_parameter(
            Name=full_parameter_name,
            WithDecryption=True
        )
        return response['Parameter']['Value']
    except Exception as e:
        print(f"Error getting parameter {full_parameter_name}: {str(e)}")
        # If unable to get from Parameter Store, try getting from environment variables
        return os.environ.get('BUCKET_NAME')

def lambda_handler(event, context):
    """
    Lambda function to handle image retrieval requests
    
    Event format:
    {
        "image_key": "path/to/image.jpg"
    }
    """
    try:
        # Get bucket name from environment variables
        bucket_name = os.environ['BUCKET_NAME']
        
        # Get parameters
        image_key = event.get('image_key')
        
        if not image_key:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required parameter: image_key'
                })
            }
        
        # Add users/ prefix
        if not image_key.startswith('users/'):
            image_key = f'users/{image_key}'
            
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Generate presigned URL (valid for 1 hour)
        presigned_url = s3_client.generate_presigned_url('get_object',
                                                        Params={'Bucket': bucket_name,
                                                               'Key': image_key},
                                                        ExpiresIn=3600)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'url': presigned_url,
                'expires_in': 3600
            })
        }
        
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return {
                'statusCode': 404,
                'body': json.dumps({
                    'error': 'Image not found'
                })
            }
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        } 