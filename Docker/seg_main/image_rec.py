import os
import pickle
import cv2
import numpy as np
from skimage import color
import statsmodels.api as sm
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift
from collections import Counter
from scipy.spatial import cKDTree
import logging
import traceback

logger = logging.getLogger()

class GaussianModel:
    def __init__(self, weights, means, covariances):
        """
        Initialize a GaussianModel with given weights, means, and covariances.
        
        Parms:
        weights: 1D array-like, model weights.
        means: 2D array-like, means of Gaussian components.
        covariances: 3D array-like, covariances of Gaussian components.
        """
        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        
    def score_samples(self, X):
        """
        Compute log probability under the model for each sample in X.
        
        Args:
            X: 2D array-like, input data.
        
        Returns:
            log_prob: 1D array-like, log probabilities of each sample under the model.
        """
        return np.array([self._multivariate_gaussian_pdf(x) for x in X])

    def _multivariate_gaussian_pdf(self, X):
        """
        Compute the probability density function of multivariate Gaussian distribution at X.
        
        Args:
            X: 1D array-like, a "row vector" or "column vector".
        
        Returns:
            pdf: float, the probability density at X under the distribution.
        """
        size = len(X)
        det = np.linalg.det(self.covariances_[0])
        norm_const = 1.0 / (np.power((2*np.pi),float(size)/2) * np.power(det,1.0/2))
        x_mu = np.matrix(X - self.means_[0])
        inv = np.linalg.inv(self.covariances_[0])
        result = np.power(np.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result


class ColorMatrix:
    """
    Class for creating and managing a matrix representation of the distances between colors.
    """
    def __init__(self, colors):
        """
        Initializes the ColorMatrix with a set of colors.

        Args:
            colors (list or ndarray): List of RGB colors.
        """
        self.colors = np.array(colors)
        self.dist_matrix = np.zeros((len(colors), len(colors)))

    @staticmethod
    def compute_color_distance(color1, color2):
        """
        Computes the Euclidean distance between two colors.

        Args:
            color1 (ndarray): First RGB color.
            color2 (ndarray): Second RGB color.

        Returns:
            float: Euclidean distance between the two colors.
        """
        distance = np.linalg.norm(color1 - color2)
        return distance

    def generate_matrix(self):
        """
        Generates a matrix of color distances. The element at [i,j] contains
        the distance between color i and color j from the input colors.

        Returns:
            ndarray: Matrix of color distances.
        """
        for i in range(len(self.colors)):
            for j in range(len(self.colors)):
                self.dist_matrix[i, j] = self.compute_color_distance(self.colors[i], self.colors[j])
        return self.dist_matrix
    
    def print_to_file(self, output_path):
        """
        Writes the first row of the distance matrix to an output file.

        Args:
            output_path (str): Path to the output file.
        """
        np.savetxt(output_path, np.floor(self.dist_matrix[0,:]), fmt='%.2f')
    

class Normalization:
    def __init__(self, img, label):
        self.img = img
        self.label = label
        self.xyz_img = color.rgb2xyz(self.img)
        self.specify = (0.524843726, 0.46181434756722176, 0.4467575836149937) # 0.76919, 0.82642, 0.817095     0.47615657092855585, 0.3650081302097927, 0.2708999766702957
    
    def extract_background_color(self):
        img_xyz = self.xyz_img
        self.img_xyz_flat = img_xyz.reshape(-1, 3)
        label_flat = self.label.flatten()
        new_list = []
        
        for i in range(len(label_flat)):
            if label_flat[i] == -1:
                new_list.append(self.img_xyz_flat[i])
                
        return np.array(new_list)
    
    def calculate_XYZ(self):
        xyz_array = self.extract_background_color()
        self.mean_xyz = xyz_array.mean(axis=0)
    
    def reconstruct(self):
        self.calculate_XYZ()
        t_add = self.specify - np.array(self.mean_xyz)
        t_rec = self.img_xyz_flat + t_add
        t_rec_rgb = color.xyz2rgb(t_rec)
        t_rec_rgb = (t_rec_rgb * 255).astype(np.uint8)
        debug = (self.img).reshape(-1, (self.img).shape[-1])
        return t_rec_rgb

class Identification:
    """
    A class for identifying and analyzing image segments based on their color properties.
    """
    def __init__(self, img_1d, label_1d):
        """
        Initializes the Identification with image data and its corresponding labels.

        Args:
            img_1d (ndarray): 1D representation of the image.
            label_1d (ndarray): 1D representation of the labels.
        """
        try:
            self.img_1d = img_1d
            self.label_1d = label_1d
            self.unique_label = np.unique(self.label_1d)
            self.data = np.array(self.all_gradient())
        except Exception as e:
            logger.error(f"Error initializing Identification: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def config_feature(self, material):
        """
        This method is used to generate a txt document for new materials.

        Args:
            material (str): Name of the material.
        """
        gradient_matrix = self.data
        flack_label = self.unique_label[(self.unique_label != -1)]
        save_matrix = (np.vstack((flack_label, gradient_matrix)))
        user_input1 = input("Do you want to save the data? (yes/no): ")
        if user_input1.lower() == 'yes':
            if os.stat(f'config/{material}_config.txt').st_size == 0:
                with open(f'config/{material}_config.txt', 'a') as f:
                    np.savetxt(f, save_matrix[0:1].astype(int), fmt='%-11d')
                    np.savetxt(f, save_matrix[1:], fmt='%-11.6f')
            else:
                existing_matrix = np.loadtxt(f'config/{material}_config.txt')
                combined_matrix = np.hstack((existing_matrix, save_matrix))
                with open(f'config/{material}_config.txt', 'w') as f:
                    np.savetxt(f, combined_matrix[0:1].astype(int), fmt='%-11d')
                    np.savetxt(f, combined_matrix[1:], fmt='%-11.6f')
                
        user_input2 = input("Do you want to generate sav model? (yes/no): ")
        if user_input2.lower() == 'yes':
            sav_filename = f'model/{material}.sav'
            data = np.loadtxt(f'config/{material}_config.txt')
            labels = data[0]
            X = np.column_stack((data[1], data[3], data[5]))
            X = sm.add_constant(X)
            model = sm.OLS(labels, X)
            fitting = model.fit()
            pickle.dump(fitting, open(sav_filename, 'wb'))
        else:
            print("Operation cancelled by the user.")
            
    
    def test_feature(self, material):
        """
        Tests the features of a given material using a saved model.

        Args:
            material (str): Name of the material.

        Returns:
            ndarray: Predicted labels for the segments.
        """
        try:
            data = self.data
            sav_filename = f'model/{material}.sav'
            
            try:
                loaded_model = pickle.load(open(sav_filename, 'rb'))
            except Exception as e:
                logger.error(f"Failed to load model for material {material}: {str(e)}")
                raise
                
            lab = self.label_1d.copy()
            for i in range(len(data[0])):
                x1 = data[0][i]
                x2 = data[2][i]
                x3 = data[4][i]
                inputs = np.array([1, x1, x2, x3]).reshape(1, -1)
                predicted_label = loaded_model.predict(inputs)
                predicted_label_rounded = abs(np.round(predicted_label))
                predicted_label_rounded = -1 if predicted_label_rounded == 0 else predicted_label_rounded
                lab[self.label_1d == i] = predicted_label_rounded
            return lab
        except Exception as e:
            logger.error(f"Error in test_feature for material {material}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def clean_huge_variance(self):
        """
        Removes segments with large color variances, treating them as noise.

        Returns:
            ndarray: Cleaned up labels for the segments.
        """
        data = self.data
        return_label = (self.label_1d).copy()
        clean_huge_variance = np.where(data[1, :] > (data[0, :]*0.8))[0]
        try:
            if len(clean_huge_variance)>0:
                mask = np.isin(self.label_1d, clean_huge_variance)
                pixels = self.img_1d[mask]
                cluster_centers, dbscan_label, dbscan_unique_labels = dbscan(pixels, eps=1, min_samples=3)
                _, mean_shift_labels, _ = mean_shift(np.array(cluster_centers), dbscan_label, dbscan_unique_labels, bandwidth=7)
                means = label_mean(pixels.reshape(-1, 3), mean_shift_labels)
                final_label = final_fitting(pixels.reshape(-1, 3), means)
                # final_label =clustering.final_fitting_fast(pixels.reshape(-1, 3), means)
                detail_label = [i + len(self.unique_label) for i in final_label]
                return_label[mask] = detail_label
                return self.relabel(return_label)
            else:
                return self.label_1d
            
        except Exception as e:
            return self.label_1d
        
    def all_gradient(self):
        """
        Computes the gradients of different color spaces (RGB, LAB, HSV) for the segments.

        Returns:
            tuple of ndarrays: Gradients of the mean and variance for each color space.
        """
        img_rgb = self.img_1d
        img_lab = color.rgb2lab(self.img_1d)
        img_hsv = color.rgb2hsv(self.img_1d)
        features = np.zeros((len(self.unique_label), 18))
        for i, label in enumerate(self.unique_label):
            indices = np.where(self.label_1d == label)[0]
            rgb_mean = img_rgb[indices].mean(axis=0)
            lab_mean = img_lab[indices].mean(axis=0)
            hsv_mean = img_hsv[indices].mean(axis=0)
            rgb_var = img_rgb[indices].var(axis=0)
            lab_var = img_lab[indices].var(axis=0)
            hsv_var = img_hsv[indices].var(axis=0)
            features[i, :] = np.concatenate([rgb_mean, rgb_var, lab_mean, lab_var, hsv_mean, hsv_var])

        rgb_mean_gradient = (ColorMatrix(features[:,:3]).generate_matrix())[1:,0]
        rgb_var_gradient = (ColorMatrix(features[:,3:6]).generate_matrix())[1:,0]
        lab_mean_gradient = (ColorMatrix(features[:,6:9]).generate_matrix())[1:,0]
        lab_var_gradient = (ColorMatrix(features[:,9:12]).generate_matrix())[1:,0]
        hsv_mean_gradient = (ColorMatrix(features[:,12:15]).generate_matrix())[1:,0]
        hsv_var_gradient = (ColorMatrix(features[:,15:18]).generate_matrix())[1:,0]
        return rgb_mean_gradient, rgb_var_gradient, lab_mean_gradient, lab_var_gradient, hsv_mean_gradient, hsv_var_gradient
    
    def relabel(self, label):
        """
        Relabels the segments to have consecutive integers starting from 0.

        Args:
            label (ndarray): Original labels.

        Returns:
            ndarray: Relabeled segments.
        """
        unique_labels = np.unique(label)
        mapping = {}
        new_label = -1
        for old_label in unique_labels:
            if old_label == -1:
                mapping[old_label] = old_label
            else:
                new_label += 1
                mapping[old_label] = new_label
        vectorized_map = np.vectorize(mapping.get)
        return vectorized_map(label)
    
def gmm_fitting(flat_image, labels, visualization=False):
    """
    Create clusters of points based on the labels and fit a GaussianModel to each cluster.
    
    Args:
        flat_image: 2D array-like, flattened image data.
        labels: 1D array-like, labels assigned to each point in the flattened image.
        visualization: bool, optional (default=False). If True, visualize the fitted GaussianModels.
    
    Returns:
        gmms: dict, a mapping from label to the fitted GaussianModel for that cluster.
    """
    #  create a cluster dictionary
    clusters = {}

    for point, label in zip(flat_image, labels):
        if label == -1:
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(point)

    for label in clusters:
        clusters[label] = np.array(clusters[label])

    gmms = {}
    for label, points in clusters.items():
        mean = np.mean(points, axis=0)
        cov = np.cov(points.T)
        gmm = GaussianModel(weights=np.array([1.0]), means=mean[np.newaxis, :], covariances=cov[np.newaxis, :, :])
        gmms[label] = gmm

    return gmms

def label_mean(flat_image, labels):
    """
    Compute the mean of color values for each unique label and return a dictionary mapping labels to their means.

    Args:
        flat_image (numpy.ndarray): Flattened image data with color values.
        labels (numpy.ndarray): Labels assigned to each point in the flattened image.

    Returns:
        dict: A dictionary mapping each unique label to its mean color value.
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    clusters = {}
    for label in unique_labels:
        points = flat_image[labels == label]
        clusters[label] = list(map(int, np.mean(points, axis=0)))
        # clusters[label] = np.mean(clusters[label])
    return clusters

def dbscan(resized_image, eps, min_samples):
    """
    Apply DBSCAN clustering on a resized image to find clusters based on RGB values and return 
    the cluster centers, labels assigned to each pixel, and unique labels.
    
    Args:
        resized_image (numpy.ndarray): 2D image data with RGB values.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

    Returns:
        tuple: A tuple containing:
            - list of numpy arrays: Mean RGB values for each cluster.
            - numpy.ndarray: Labels assigned to each pixel in the flattened image.
            - numpy.ndarray: Unique labels for each cluster, excluding the -1 label for outliers.
    """
    try:
        flat_image_rgb = resized_image.reshape(-1, 3)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='auto').fit(flat_image_rgb)
        labels = dbscan.labels_
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # Exclude the -1 label for outliers
        cluster_centers = [flat_image_rgb[labels == label].mean(axis=0) for label in unique_labels]
        return cluster_centers, labels, unique_labels
    except Exception as e:
        logger.error(f"Error in DBSCAN clustering: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def mean_shift(clusters, labels, unique_labels,show = False, bandwidth=10):
    """
    Function to perform MeanShift clustering

    Args:
        clusters (ndarray): The dataset on which to perform clustering.
        labels (ndarray): The labels assigned to each data point.
        unique_labels (ndarray): The unique labels in the dataset.

    Returns:
        labels (ndarray): The updated labels assigned to each data point.
        unique_labels (ndarray): The updated unique labels in the dataset.
        cluster_centers (ndarray): The updated values of the cluster centers.
    """
  
    # After training the model, we store the coordinates for the cluster centers
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(clusters)
    cluster_centers = ms.cluster_centers_
    
    # Assign each data point to the closest cluster center
    new_labels = np.argmin(distance.cdist(clusters, cluster_centers), axis=1)
    
    # Update unique labels
    unique_labels_new = np.unique(new_labels)

    # Print out mapping from old labels to new labels
    label_mapping = dict(zip(unique_labels, new_labels))
    labels = np.array([label_mapping.get(label, -1) for label in labels])
    return cluster_centers, labels, unique_labels_new

def nearest_label(points_batch, kdtree, labels):
    """
    For each point in the points_batch, find the nearest label based on the KDTree.

    Args:
        points_batch (list[numpy.ndarray]): A batch of points.
        kdtree (cKDTree): KDTree constructed based on mean values.
        labels (list): List of labels corresponding to the mean values.

    Returns:
        list: A list of labels corresponding to each point in the points_batch.
    """
    _, indices = kdtree.query(points_batch)
    return [labels[index] for index in indices]

def final_fitting(img, means, batch_size= 1000):
    """
    为图像中的每个像素基于最近的均值分配标签。
    
    参数:
        img (ndarray): 输入图像,形状为 (height, width, channels)
        means (dict): 字典,键为标签,值为对应的均值
        batch_size (int): 已不再使用的参数,保留用于向后兼容
        
    返回:
        ndarray: 图像中每个像素的标签分配结果
    """
    flat_img = img.reshape(-1, img.shape[-1])
    labels, values = zip(*means.items())
    values = [np.array(value).astype(np.float32) for value in values]
    kdtree = cKDTree(values)
    
    # 直接处理所有像素,不再分批
    _, indices = kdtree.query(flat_img)
    final_labels = np.array([labels[index] for index in indices])
    
    return final_labels

def final_fitting_fast(img, means):
    """
    Convert image to gray scale image and assign each pixel a label based on the closest mean value.

    Args:
        img (numpy.ndarray): The input image with shape (height, width, channels).
        means (dict): Dictionary with labels as keys and corresponding mean values (possibly list or array).

    Returns:
        ndarray: Flattened label assignments for each pixel in the image.
    """
    for label in means.keys():
        means[label] = np.mean(means[label])
    flat_img = img.reshape(-1, img.shape[-1])
    flat_img = np.mean(flat_img, axis=1)
    
    labels = np.array(list(means.keys()))
    mean_values = np.array(list(means.values()))
    
    differences = np.abs(flat_img[:, np.newaxis] - mean_values)
    min_diff_idx = np.argmin(differences, axis=1)
    final_labels = labels[min_diff_idx]
    return final_labels

def get_common_clusters(final_label, img, n_clusters=2):
    """
    Given the final labels and original image, return the 'n_clusters' most common cluster colors.

    Args:
        final_label (list): List of final labels for each point.
        img (ndarray): Original image as a numpy array.
        n_clusters (int, optional): Number of common clusters to return. Defaults to 2.
        
    Returns:
        List of the most common cluster colors.
    """
    def check_bg(gradient_matrix, t3_labels):
        bg_labels = []
        bg_labels.append(t3_labels[0])
        for i in range(n_clusters):
            if gradient_matrix[0,i]<=18:
                bg_labels.append(t3_labels[i])
        return bg_labels
    
    flat_labels = final_label.flatten()
    flat_img = img.reshape(-1, img.shape[-1])
    
    label_counts = Counter(flat_labels)
    common_labels = label_counts.most_common()
    
    # Compute average brightness for each common label
    brightnesses = {}
    for label, _ in common_labels:
        indices = np.where(flat_labels == label)
        cluster_color = flat_img[indices].mean(axis=0)
        brightness = cluster_color.mean()  # Average RGB values
        brightnesses[label] = brightness

    # Sort common_labels by count and brightness
    common_labels = sorted(common_labels, key=lambda x: (x[1], brightnesses[x[0]]), reverse=True)
    
    # Select the top n_clusters
    common_labels = common_labels[:n_clusters]

    if len(common_labels) == 1:
        final_label[final_label == common_labels[0][0]] = -1
        return final_label

    t3_labels = np.array(common_labels)[:,0]
    common_cluster_colors = []
    for label, _ in common_labels:
        indices = np.where(flat_labels == label)
        cluster_color = flat_img[indices].mean(axis=0)
        common_cluster_colors.append(cluster_color)
        
    color_matrix = ColorMatrix(common_cluster_colors)
    gradient_matrix = color_matrix.generate_matrix()
    bg_labels = check_bg(gradient_matrix, t3_labels)
    
    for label in bg_labels:
        final_label[final_label == label] = -1

    # Now renumber the remaining labels
    unique_labels = np.unique(final_label)
    # Ignore the -1 label
    unique_labels = unique_labels[unique_labels != -1]
    
    # Now re-number the labels from 0
    for new_label, old_label in enumerate(unique_labels):
        final_label[final_label == old_label] = new_label

    return final_label

def clean_huge_area(label, input_image):
    """
    Remove huge and tiny areas from the label by processing based on the label value.
    
    Args:
        label (ndarray): 1D array containing labels for the flattened image.
        input_image (ndarray): 2D image data.
    
    Returns:
        numpy.ndarray: Cleaned 1D label array after processing.
    """
    def process_label(input_label, value):
        """
        Process labels to remove huge and tiny areas based on the given value.

        Args:
            input_label (ndarray): 2D array of label values.
            value (int): Label value to be processed.

        Returns:
            numpy.ndarray: Processed 2D label array.
        """
        binary_label = np.where(input_label == value, 1, 0).astype(np.uint8)
        mask = binary_label.reshape(input_image.shape[:2])
        num_labels_pre, labels_im_pre = cv2.connectedComponents(mask)
        unique, counts = np.unique(labels_im_pre, return_counts=True)
        remove_small_area = np.isin(labels_im_pre, unique[counts < 10])
        mask[remove_small_area] = 0
        kernel = np.ones((3,3),np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations = 1)
        num_labels, labels_im = cv2.connectedComponents(dilated_mask)
        unique, counts = np.unique(labels_im, return_counts=True)
        remove_large_area = np.isin(labels_im, unique[counts > 0.15 * total_pixels])
        mask[remove_large_area] = 0
        
        m = np.logical_and(mask == 0, input_label == value)
        input_label[m] = -1
        return input_label
    
    original_image = input_image.copy()
    height, width = original_image.shape[:2]
    total_pixels = height * width
    label_2D = label.reshape(original_image.shape[:2])
    
    label_2D = process_label(label_2D, 1)
    label_2D = process_label(label_2D, 2)
    
    return label_2D.reshape(-1, 1)
