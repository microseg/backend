import numpy as np
import cv2
import json
import boto3

def get_image_from_s3(bucket_name, image_key):
    s3_client = boto3.client('s3')
    
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
        image_content = response['Body'].read()
        nparr = np.frombuffer(image_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    except Exception as e:
        print(f"Error reading from S3: {str(e)}")
        raise

def lambda_handler(event, context):
    try:
        if 'body' in event:
            body = json.loads(event['body'])
            bucket_name = body.get('bucket_name')
            image_key = body.get('image_key')
            material = body.get('material', 'Graphene')
        else:
            bucket_name = event.get('bucket_name')
            image_key = event.get('image_key')
            material = event.get('material', 'Graphene')
            
        if not bucket_name or not image_key:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing bucket_name or image_key'})
            }
            
        img = get_image_from_s3(bucket_name, image_key)
        test_img_size = 256
        resized_img = reshape_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), test_img_size)
        
        result_encoded = resized_img.tolist()
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'original_resized': result_encoded,
                'material': material,
                'image_processed': image_key
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }

def reshape_image(img, short_edge_length):
    '''
    Function to reshape image base on the short edge length
    
    Args:
        img (ndarray): The image to be reshaped.
        short_edge_length (int): The number of pixels for the shortest edge.

    Returns:
        resized_img (ndarray): The reshaped image.
    '''
    height, width = img.shape[:2]

    # Identify the short edge and calculate scale
    if height < width:
        scale = short_edge_length / height
        new_height = short_edge_length
        new_width = int(scale * width)
    else:
        scale = short_edge_length / width
        new_width = short_edge_length
        new_height = int(scale * height)

    # Resize the image
    resized_img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR)
    return resized_img
    