import numpy as np
import cv2
import json
import time
import boto3
from image_rec import *
from matplotlib import cm
from pre_utils import ImagePreprocessor

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
        segmenter = Material_seg(img, material)
        result = segmenter.merged_list()
        result_encoded = result.tolist()
        return {
            'statusCode': 200,
            'body': json.dumps({
                'result': result_encoded,
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
        st_0 = time.time()
        self.original_image = self.preprocessor.process(self.original_image)
        img = change_col_space(self.original_image, color_space) 
        resized_image = reshape_image(img, 64)
        flat_image = resized_image.reshape(-1, 3)
        t_0 = int((time.time()-st_0)*1000)
        
        st_1 = time.time()
        cluster_centers, dbscan_label, dbscan_unique_labels = dbscan(resized_image, eps=1, min_samples=3)
        mean_shift_cluster_centers, mean_shift_labels, _ = mean_shift(np.array(cluster_centers), dbscan_label, dbscan_unique_labels, bandwidth=7)
        means = label_mean(flat_image, mean_shift_labels)
        t_1 = int((time.time()-st_1)*1000)
        
        st_2 = time.time()
        test_img_size = 256
        resized_image = reshape_image(img, test_img_size)
        flat_image = resized_image.reshape(-1, 3)
        final_label = final_fitting(resized_image, means) # most time consumption
        final_label = get_common_clusters(final_label, resized_image,n_clusters=3)
        t_2 = int((time.time()-st_2)*1000)
        
        st_3 = time.time()
        self.original_img_backup_resized = reshape_image(cv2.cvtColor(self.original_image_backup, cv2.COLOR_BGR2RGB), test_img_size)
        rec_img = (self.original_img_backup_resized).reshape(-1, (self.original_img_backup_resized).shape[-1])
        re_cluster_px = Identification(rec_img, final_label).clean_huge_variance() 
        t_3 = int((time.time()-st_3)*1000)
        
        st_4 = time.time()
        label = Identification(rec_img, re_cluster_px).test_feature(self.material)
        t_4 = int((time.time()-st_4)*1000)

        
        print(f"image preprocess time: {t_0} ms\ncore cluster time: {t_1} ms\nfine cluster time: {t_2} ms\n2nd cluster time: {t_3} ms\nlayer identification time: {t_4} ms")

            
        return label

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
    