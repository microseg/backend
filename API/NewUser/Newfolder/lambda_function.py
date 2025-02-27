import json
import boto3
import os
from botocore.exceptions import ClientError

def lambda_handler(event, context):
    """
    Lambda函数处理用户文件夹创建请求
    
    事件格式:
    {
        "user_id": "sub_from_cognito"
    }
    """
    try:
        # 从环境变量获取bucket名称
        bucket_name = os.environ['BUCKET_NAME']
        
        # 获取用户ID
        user_id = event.get('user_id')
        
        if not user_id:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required parameter: user_id'
                })
            }
            
        # 初始化S3客户端
        s3_client = boto3.client('s3')
        
        # 创建用户文件夹（在S3中，文件夹是通过以'/'结尾的对象键来表示的）
        folder_key = f'users/{user_id}/'
        
        # 创建一个空对象来表示文件夹
        s3_client.put_object(
            Bucket=bucket_name,
            Key=folder_key
        )
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'message': 'User folder created successfully',
                'folder_key': folder_key
            })
        }
        
    except ClientError as e:
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