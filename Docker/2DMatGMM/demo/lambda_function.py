"""
Lambda function for 2DMatGMM material detection.
Process a single image and return detection results in JSON format.
"""
import json
import os
import logging
import traceback
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from botocore.exceptions import ClientError

import cv2
import numpy as np
import boto3

from demo.demo_functions import visualise_flakes
from GMMDetector import MaterialDetector
from demo.rle import RLE

# Configure logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Error codes mapping
ERROR_CODES = {
    'InvalidParameterError': 400,
    'AccessDenied': 403,
    'ResourceNotFound': 404,
    'InternalError': 500,
    'S3OperationError': 502
}

@dataclass
class DetectionParameters:
    """Detection parameters configuration."""
    material: str = "Graphene"
    size_threshold: int = 500
    std_threshold: float = 5.0
    min_confidence: float = 0.6

class S3Handler:
    """Handler class for S3 operations."""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
    
    def get_image(self, bucket_name: str, image_key: str) -> np.ndarray:
        """
        Read image from S3 bucket.
        
        Args:
            bucket_name (str): Name of the S3 bucket
            image_key (str): Key of the image in the bucket
            
        Returns:
            np.ndarray: Image as numpy array
        """
        try:
            response = self.s3_client.get_object(Bucket=bucket_name, Key=image_key)
            image_content = response['Body'].read()
            
            nparr = np.frombuffer(image_content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError(f"Failed to decode image from S3: {image_key}")
            
            return img
        except Exception as e:
            error_msg = f"Error reading image from S3: bucket={bucket_name}, key={image_key}, error={str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def save_results(self, bucket_name: str, image_key: str, results: Dict[str, Any], 
                    image: np.ndarray) -> Tuple[str, str]:
        """
        Save results to S3 bucket.
        
        Args:
            bucket_name (str): Name of the S3 bucket
            image_key (str): Original image key (e.g. 'users/user-uuid/image.jpg')
            results (Dict[str, Any]): Results to save
            image (np.ndarray): Processed image to save
            
        Returns:
            Tuple[str, str]: (JSON key in S3, Image key in S3)
        """
        try:
            user_folder = os.path.dirname(image_key)  # Example: users/user-uuid
            base_name = os.path.splitext(os.path.basename(image_key))[0]  # Example: Graphene_27
            
            results_folder = f"{user_folder}/__results"  # Example: users/user-uuid/__results
            json_key = f"{results_folder}/{base_name}_results.json"
            image_key = f"{results_folder}/{base_name}_processed.jpg"
            
            json_bytes = json.dumps(results, indent=2).encode('utf-8')
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=json_key,
                Body=json_bytes,
                ContentType='application/json'
            )
            
            _, img_encoded = cv2.imencode('.jpg', image)
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=image_key,
                Body=img_encoded.tobytes(),
                ContentType='image/jpeg'
            )
            
            return json_key, image_key
        except Exception as e:
            error_msg = f"Error saving results to S3: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

class FlakeDetector:
    """Main class for flake detection and processing."""
    
    def __init__(self, params: DetectionParameters):
        """
        Initialize FlakeDetector.
        
        Args:
            params (DetectionParameters): Detection parameters
        """
        self.params = params
        self.file_dir = os.path.dirname(os.path.abspath(__file__))
        self.model = self._initialize_model()
        
    def _initialize_model(self) -> MaterialDetector:
        """Initialize the MaterialDetector model."""
        contrast_path_root = os.path.join(self.file_dir, "..", "GMMDetector", "trained_parameters")
        fp_detector_path = os.path.join(self.file_dir, "..", "FalsePositiveDetector", "models", "classifier_L2_logistic.joblib")
        
        # Load GMM parameters
        with open(os.path.join(contrast_path_root, f"{self.params.material}_GMM.json")) as f:
            contrast_dict = json.load(f)
        
        return MaterialDetector(
            contrast_dict=contrast_dict,
            size_threshold=self.params.size_threshold,
            standard_deviation_threshold=self.params.std_threshold,
            used_channels="BGR",
            false_positive_detector_path=fp_detector_path
        )
    
    def _process_flake(self, flake) -> Dict[str, Any]:
        """Process a single flake and convert to dictionary format."""
        flake_dict = flake.to_dict()
        
        # Convert mask to RLE format
        if isinstance(flake_dict["mask"], np.ndarray):
            rle = RLE.from_mask(flake_dict["mask"])
            flake_dict["mask"] = rle.to_dict()
        
        # Convert other numpy arrays to lists
        if isinstance(flake_dict["mean_contrast"], np.ndarray):
            flake_dict["mean_contrast"] = flake_dict["mean_contrast"].tolist()
        else:
            flake_dict["mean_contrast"] = list(flake_dict["mean_contrast"])
        
        flake_dict["center"] = list(flake_dict["center"])
        return flake_dict
    
    def process_image(self, image: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Process an image and detect flakes.
        
        Args:
            image (np.ndarray): Input image as numpy array
            
        Returns:
            Tuple[Dict[str, Any], np.ndarray]: (Detection results, Processed image)
        """
        # Detect flakes
        flakes = self.model(image)
        
        # Generate output image
        image_overlay = visualise_flakes(flakes, image, self.params.min_confidence)
        
        # Process flakes
        flakes_info = [
            self._process_flake(flake) 
            for flake in flakes 
            if flake.false_positive_probability <= (1 - self.params.min_confidence)
        ]
        
        # Prepare response
        response = {
            "material": self.params.material,
            "detection_parameters": {
                "size_threshold": self.params.size_threshold,
                "std_threshold": self.params.std_threshold,
                "min_confidence": self.params.min_confidence
            },
            "flakes_detected": len(flakes_info),
            "flakes": flakes_info
        }
        
        return response, image_overlay

class LambdaHandler:
    """Handler class for AWS Lambda function."""
    
    def __init__(self):
        """Initialize handler with S3 client and detector."""
        self.s3_handler = S3Handler()
    
    def handle(self, event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
        """
        Handle Lambda function invocation.
        
        Args:
            event (Dict[str, Any]): Lambda event
            context (Any, optional): Lambda context
            
        Returns:
            Dict[str, Any]: Processing results with status code
        """
        # Store file locations
        file_locations = {
            'input_image': None,
            'output_json': None,
            'output_image': None
        }
        
        try:
            # Extract parameters from event
            bucket_name = event.get('bucket_name')
            image_key = event.get('image_key')
            material = event.get('material', 'Graphene')
            
            # Store input image location
            file_locations['input_image'] = f"s3://{bucket_name}/{image_key}"
            
            if not all([bucket_name, image_key]):
                return {
                    'statusCode': 400,
                    'body': {
                        'error': 'Missing required parameters: bucket_name and image_key are required',
                        'file_locations': file_locations
                    }
                }
            
            # Initialize parameters
            params = DetectionParameters(material=material)
            
            # Initialize detector
            detector = FlakeDetector(params)
            
            try:
                # Get image from S3
                image = self.s3_handler.get_image(bucket_name, image_key)
            except ClientError as e:
                error_code = e.response['Error']['Code']
                status_code = ERROR_CODES.get(error_code, 500)
                return {
                    'statusCode': status_code,
                    'body': {
                        'error': f"S3 error: {str(e)}",
                        'code': error_code,
                        'file_locations': file_locations
                    }
                }
            except Exception as e:
                return {
                    'statusCode': 500,
                    'body': {
                        'error': f"Failed to read image: {str(e)}",
                        'file_locations': file_locations
                    }
                }
            
            # Process image
            try:
                results, processed_image = detector.process_image(image)
            except Exception as e:
                return {
                    'statusCode': 500,
                    'body': {
                        'error': f"Image processing error: {str(e)}",
                        'file_locations': file_locations
                    }
                }
            
            # Save results to S3
            try:
                json_key, processed_image_key = self.s3_handler.save_results(
                    bucket_name, image_key, results, processed_image
                )
                
                # Update file locations with output files
                file_locations.update({
                    'output_json': f"s3://{bucket_name}/{json_key}",
                    'output_image': f"s3://{bucket_name}/{processed_image_key}"
                })
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                status_code = ERROR_CODES.get(error_code, 500)
                return {
                    'statusCode': status_code,
                    'body': {
                        'error': f"Failed to save results: {str(e)}",
                        'code': error_code,
                        'file_locations': file_locations
                    }
                }
            
            # Add file locations to results
            results['file_locations'] = file_locations
            
            # Return successful response
            return {
                'statusCode': 200,
                'body': results
            }
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'statusCode': 500,
                'body': {
                    'error': f"Unexpected error: {str(e)}",
                    'file_locations': file_locations
                }
            }

def lambda_handler(event, context):
    """
    AWS Lambda function handler.
    
    Args:
        event (dict): Lambda event
        context (object): Lambda context
        
    Returns:
        dict: Processing results with status code
    """
    handler = LambdaHandler()
    return handler.handle(event, context)

if __name__ == "__main__":
    # Local test
    test_event = {
        "bucket_name": "your-bucket-name",
        "image_key": "users/d4b83418-50b1-7093-2187-ac12ab2173bc/Graphene_27.jpg",
        "material": "Graphene"
    }
    print(json.dumps(lambda_handler(test_event, None), indent=2))