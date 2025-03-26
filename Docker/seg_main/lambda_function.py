import numpy as np
import cv2
import json
import time
import boto3
import logging
import traceback
from datetime import datetime
from image_rec import *
from matplotlib import cm
from pre_utils import ImagePreprocessor

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def log_error_context(e, stage_name, **context):
    """
    Unified error logging function
    """
    error_context = {
        'error_type': type(e).__name__,
        'error_message': str(e),
        'stage': stage_name,
        'timestamp': datetime.now().isoformat(),
        **context
    }
    
    logger.error(f"Error occurred in stage: {stage_name}")
    logger.error(f"Error type: {error_context['error_type']}")
    logger.error(f"Error message: {error_context['error_message']}")
    logger.error(f"Detailed context: {json.dumps(error_context, default=str)}")
    logger.error(f"Stack trace: {traceback.format_exc()}")
    
    return error_context

def get_image_from_s3(bucket_name, image_key):
    s3_client = boto3.client('s3')
    
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=image_key)
        image_content = response['Body'].read()
        
        nparr = np.frombuffer(image_content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Image decoding failed")
            
        return img
    except Exception as e:
        error_context = log_error_context(e, 'S3 Image Reading', 
            bucket_name=bucket_name,
            image_key=image_key,
            image_content_size=len(image_content) if 'image_content' in locals() else None
        )
        raise Exception(f"Failed to read image from S3: {str(e)}") from e

def lambda_handler(event, context):
    start_time = datetime.now()
    request_id = context.aws_request_id
    execution_context = {
        'request_id': request_id,
        'start_time': start_time.isoformat(),
    }
    
    try:
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
                
            execution_context.update({
                'bucket_name': bucket_name,
                'image_key': image_key,
                'material': material
            })
        except Exception as e:
            error_context = log_error_context(e, 'Parameter Parsing', 
                event=event,
                execution_context=execution_context
            )
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Parameter parsing failed',
                    'details': error_context,
                    'request_id': request_id
                })
            }
            
        if not bucket_name or not image_key:
            error_msg = f"Missing required parameters. bucket_name: {bucket_name}, image_key: {image_key}"
            logger.error(error_msg)
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': error_msg,
                    'request_id': request_id
                })
            }
            
        # 主处理流程
        img = get_image_from_s3(bucket_name, image_key)
        execution_context['image_shape'] = img.shape if img is not None else None
        
        segmenter = Material_seg(img, material)
        result = segmenter.merged_list()
        result_encoded = result.tolist()
        
        execution_time = (datetime.now() - start_time).total_seconds()
        execution_context['execution_time'] = execution_time
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'result': result_encoded,
                'material': material,
                'image_processed': image_key,
                'execution_context': execution_context
            })
        }
        
    except Exception as e:
        error_context = log_error_context(e, 'Main Processing', 
            execution_context=execution_context
        )
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Processing failed',
                'details': error_context,
                'request_id': request_id
            })
        }

class Material_seg():

    def __init__(self, original_img, material, save_path = None):
        """
        Initialization

        Args:
            original_img (ndarray): input BGR color space image
            material (str): type of material (model) for segmentation
            save_path (str, optional): save folder for segmentation result. Defaults to None.
            stream (io, optional): debug console stream. Defaults to None.
        """
        self.original_image = original_img
        self.original_image_backup = original_img
        self.material = material
        self.save_path = save_path
        self.preprocessor = ImagePreprocessor(denoise_method='quality')  # Initialize with fast method
        
    def single_process(self, color_space):
        """
        Segments the image based on the provided color space and clustering techniques.

        Args:
            color_space (str): Color space to use for segmentation (e.g., "RGB", "HSV", "LAB").

        Returns:
            ndarray: Segmented label results for the image.
        """
        stage_context = {
            'color_space': color_space,
            'material': self.material,
            'stage_timings': {}
        }
        
        try:
            # Stage 1: Image Preprocessing
            st_0 = time.time()
            try:
                self.original_image = self.preprocessor.process(self.original_image)
                img = change_col_space(self.original_image, color_space) 
                resized_image = reshape_image(img, 64)
                flat_image = resized_image.reshape(-1, 3)
                stage_context['preprocessing'] = {
                    'original_shape': self.original_image.shape,
                    'resized_shape': resized_image.shape
                }
            except Exception as e:
                error_context = log_error_context(e, 'Preprocessing Stage', 
                    stage_context=stage_context,
                    image_shape=self.original_image.shape if self.original_image is not None else None
                )
                raise Exception(f"Preprocessing stage failed: {str(e)}") from e
            stage_context['stage_timings']['preprocessing'] = int((time.time()-st_0)*1000)

            # Stage 2: Initial Clustering
            st_1 = time.time()
            try:
                cluster_centers, dbscan_label, dbscan_unique_labels = dbscan(resized_image, eps=1, min_samples=3)
                if cluster_centers is None or len(cluster_centers) == 0:
                    raise ValueError("DBSCAN clustering failed: No cluster centers found")
                    
                mean_shift_cluster_centers, mean_shift_labels, _ = mean_shift(
                    np.array(cluster_centers), 
                    dbscan_label, 
                    dbscan_unique_labels, 
                    bandwidth=7
                )
                means = label_mean(flat_image, mean_shift_labels)
                stage_context['clustering'] = {
                    'dbscan_clusters': len(cluster_centers) if cluster_centers is not None else 0,
                    'mean_shift_clusters': len(mean_shift_cluster_centers) if mean_shift_cluster_centers is not None else 0
                }
            except Exception as e:
                error_context = log_error_context(e, 'Initial Clustering', 
                    stage_context=stage_context
                )
                raise Exception(f"Initial clustering stage failed: {str(e)}") from e
            stage_context['stage_timings']['initial_clustering'] = int((time.time()-st_1)*1000)

            # Stage 3: Fine Clustering
            st_2 = time.time()
            try:
                test_img_size = 256
                resized_image = reshape_image(img, test_img_size)
                flat_image = resized_image.reshape(-1, 3)
                final_label = final_fitting(resized_image, means)
                if final_label is None:
                    raise ValueError("Final fitting failed: No labels generated")
                    
                final_label = get_common_clusters(final_label, resized_image, n_clusters=3)
            except Exception as e:
                error_context = log_error_context(e, 'Fine Clustering Stage', 
                    stage_context=stage_context
                )
                raise Exception(f"Fine clustering stage failed: {str(e)}") from e
            stage_context['stage_timings']['fine_clustering'] = int((time.time()-st_2)*1000)

            # Stage 4: Image Reconstruction
            st_3 = time.time()
            try:
                self.original_img_backup_resized = reshape_image(
                    cv2.cvtColor(self.original_image_backup, cv2.COLOR_BGR2RGB), 
                    test_img_size
                )
                rec_img = (self.original_img_backup_resized).reshape(-1, (self.original_img_backup_resized).shape[-1])
                re_cluster_px = Identification(rec_img, final_label).clean_huge_variance()
            except Exception as e:
                error_context = log_error_context(e, 'Image Reconstruction Stage', 
                    stage_context=stage_context
                )
                raise Exception(f"Image reconstruction stage failed: {str(e)}") from e
            stage_context['stage_timings']['image_reconstruction'] = int((time.time()-st_3)*1000)

            # Stage 5: Material Feature Testing
            st_4 = time.time()
            try:
                label = Identification(rec_img, re_cluster_px).test_feature(self.material)
                if label is None:
                    raise ValueError(f"Feature testing failed for material: {self.material}")
            except Exception as e:
                error_context = log_error_context(e, 'Material Feature Testing Stage', 
                    stage_context=stage_context
                )
                raise Exception(f"Material feature testing stage failed: {str(e)}") from e
            stage_context['stage_timings']['material_feature_testing'] = int((time.time()-st_4)*1000)

            return label

        except Exception as e:
            error_context = log_error_context(e, 'Single Process Overall', 
                stage_context=stage_context
            )
            raise Exception(f"Image processing failed: {str(e)}") from e

    def merged_list(self):
        """
        Processes the image using the RGB color space and merges the results from multiple color spaces.
        
        Returns:
            ndarray: Cleaned segmentation results.
        """
        label = self.single_process("RGB")
        dst = clean_huge_area(label, self.original_img_backup_resized)
        final_label_2D = dst.reshape(self.original_img_backup_resized.shape[:2])
        design = "viridis"
        cmap = cm.get_cmap(design)

        # Create color mapping for labels
        unique_labels = np.unique(final_label_2D)
        normalized_labels = (unique_labels - np.min(unique_labels)) / (np.max(unique_labels) - np.min(unique_labels))
        color_map = {label: (np.array(cmap(norm_label)[:3]) * 255).astype(int) for label, norm_label in zip(unique_labels, normalized_labels) if label != -1}

        # Create final_label image
        final_label_img = np.zeros_like(self.original_img_backup_resized)
        for label, color in color_map.items():
            final_label_img[final_label_2D == label] = color
        final_label_img[final_label_2D == -1] = [0, 0, 0]
        return final_label_img
    
def change_col_space(img, color_space):
    # img = green_channel(img)
    color_space_conversion = {
        "RGB": cv2.COLOR_BGR2RGB,
        "LAB": cv2.COLOR_BGR2LAB,
        "HSV": cv2.COLOR_BGR2HSV,
    }
    if color_space in color_space_conversion:
        img = cv2.cvtColor(img, color_space_conversion[color_space])
    return img

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
    