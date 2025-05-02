import json
import boto3
import os
from botocore.exceptions import ClientError
from typing import Dict, List

class UserFolderManager:
    """
    A class to manage the creation of user folders in S3 bucket
    """
    
    def __init__(self, bucket_name: str, user_id: str):
        """
        Initialize the UserFolderManager
        
        Args:
            bucket_name (str): Name of the S3 bucket
            user_id (str): Unique identifier for the user
        """
        self.bucket_name = bucket_name
        self.user_id = user_id
        self.s3_client = boto3.client('s3')
        self.base_folder_key = f'users/{user_id}/'
        
    def get_folder_structure(self) -> Dict[str, str]:
        """
        Define the folder structure for the user
        
        Returns:
            Dict[str, str]: Dictionary containing folder paths
        """
        return {
            'base_folder': self.base_folder_key,
            'original_images': f'{self.base_folder_key}Original Images/',
            'tag_images': f'{self.base_folder_key}Tag Images/',
            'result': f'{self.base_folder_key}__result/'
        }
    
    def create_folders(self) -> None:
        """
        Create all required folders in S3 bucket
        """
        folders = self.get_folder_structure().values()
        for folder in folders:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=folder
            )
    
    def generate_response(self, status_code: int, message: str, is_error: bool = False) -> Dict:
        """
        Generate API response
        
        Args:
            status_code (int): HTTP status code
            message (str): Response message
            is_error (bool): Whether this is an error response
            
        Returns:
            Dict: Formatted API response
        """
        body = {
            'message': message
        }
        
        if is_error:
            body['error'] = message
        else:
            body['folders'] = self.get_folder_structure()
            
        return {
            'statusCode': status_code,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(body),
            'isBase64Encoded': False
        }

    def get_client_error_status_code(self, error_code: str) -> int:
        """
        Map AWS error codes to appropriate HTTP status codes
        
        Args:
            error_code (str): AWS error code
            
        Returns:
            int: Corresponding HTTP status code
        """
        error_code_mapping = {
            'NoSuchBucket': 404,  # Not Found
            'AccessDenied': 403,  # Forbidden
            'InvalidToken': 401,  # Unauthorized
            'ThrottlingException': 429,  # Too Many Requests
            'ValidationError': 400,  # Bad Request
            'MalformedPolicy': 400,  # Bad Request
            'InvalidParameter': 400,  # Bad Request
        }
        return error_code_mapping.get(error_code, 500)  # Default to 500 if code not mapped

def lambda_handler(event: Dict, context) -> Dict:
    """
    AWS Lambda handler for user folder creation
    
    Args:
        event (Dict): Lambda event containing user_id
        context: Lambda context
        
    Returns:
        Dict: API response with status and folder information
    """
    try:
        # Get bucket name from environment variables
        bucket_name = os.environ['BUCKET_NAME']
        
        # Get and validate user_id from event
        user_id = event.get('user_id')
        if not user_id:
            return UserFolderManager(bucket_name, '').generate_response(
                status_code=400,
                message='Missing required parameter: user_id',
                is_error=True
            )
        
        # Initialize folder manager and create folders
        folder_manager = UserFolderManager(bucket_name, user_id)
        folder_manager.create_folders()
        
        # Return success response
        return folder_manager.generate_response(
            status_code=200,
            message='User folders created successfully'
        )
        
    except ClientError as e:
        folder_manager = UserFolderManager(bucket_name, user_id if 'user_id' in locals() else '')
        error_code = e.response['Error']['Code']
        status_code = folder_manager.get_client_error_status_code(error_code)
        error_message = f'AWS Error ({error_code}): {e.response["Error"]["Message"]}'
        
        return folder_manager.generate_response(
            status_code=status_code,
            message=error_message,
            is_error=True
        )
    except KeyError as e:
        # Handle missing environment variables or configuration
        folder_manager = UserFolderManager('', '')
        return folder_manager.generate_response(
            status_code=500,
            message=f'Configuration Error: Missing {str(e)}',
            is_error=True
        )
    except Exception as e:
        # Handle unexpected errors
        folder_manager = UserFolderManager(bucket_name if 'bucket_name' in locals() else '', 
                                         user_id if 'user_id' in locals() else '')
        return folder_manager.generate_response(
            status_code=500,
            message=f'Internal Server Error: {str(e)}',
            is_error=True
        ) 