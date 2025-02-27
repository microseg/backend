import cv2

class ImagePreprocessor:
    """
    A class for preprocessing images with configurable enhancement techniques.
    This class provides a pipeline for image preprocessing with various configurable methods.
    
    Attributes:
        denoise_method (str): Method for denoising:
            - 'none': No denoising
            - 'fast': Gaussian blur (fastest but may blur edges)
            - 'medium': Median blur (good for salt-and-pepper noise)
            - 'quality': Non-local means denoising (best quality but slowest)
        max_eq_pixels (int): Maximum number of pixels for applying histogram equalization
        bilateral_d (int): Diameter of each pixel neighborhood in bilateral filter
        bilateral_sigma_color (float): Filter sigma in the color space (larger values mix colors)
        bilateral_sigma_space (float): Filter sigma in the coordinate space (larger values mix farther pixels)
    """
    
    def __init__(self, 
                 denoise_method='quality',
                 max_eq_pixels=1200000,
                 bilateral_d=16,
                 bilateral_sigma_color=40,
                 bilateral_sigma_space=40):
        """
        Initialize the ImagePreprocessor with specified parameters.
        
        Args:
            denoise_method (str): Denoising method to use ('none', 'fast', 'medium', 'quality')
            max_eq_pixels (int): Maximum pixels for histogram equalization (default: 1.2M pixels)
            bilateral_d (int): Diameter of each pixel neighborhood (default: 16)
            bilateral_sigma_color (float): Filter sigma in color space (default: 40)
            bilateral_sigma_space (float): Filter sigma in coordinate space (default: 40)
        """
        self.denoise_method = denoise_method
        self.max_eq_pixels = max_eq_pixels
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
    
    def apply_histogram_equalization(self, img):
        """
        Apply histogram equalization to each channel if image is small enough.
        This improves contrast by effectively spreading out the most frequent intensity values.
        Only applied to images smaller than max_eq_pixels to avoid processing large images.
        
        Args:
            img (numpy.ndarray): Input BGR image
            
        Returns:
            numpy.ndarray: Image with equalized histogram or original if too large
        """
        height, width, _ = img.shape
        if width * height <= self.max_eq_pixels:
            channels = cv2.split(img)
            eq_channels = [cv2.equalizeHist(channel) for channel in channels]
            return cv2.merge(eq_channels)
        return img
    
    def apply_denoising(self, img):
        """
        Apply the selected denoising method to the image.
        Different methods offer trade-offs between speed and quality:
        - 'fast': Uses Gaussian blur, fastest but may blur edges
        - 'medium': Uses median blur, good for salt-and-pepper noise
        - 'quality': Uses non-local means denoising, best quality but slowest
        - 'none': No denoising applied
        
        Args:
            img (numpy.ndarray): Input BGR image
            
        Returns:
            numpy.ndarray: Denoised image
        """
        if self.denoise_method == 'fast':
            return cv2.GaussianBlur(img, (5, 5), 0)
        elif self.denoise_method == 'medium':
            return cv2.medianBlur(img, 5)
        elif self.denoise_method == 'quality':
            return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 17)
        return img  # 'none' case
    
    def auto_contrast(self, rgb):
        """
        Adjust the contrast of an image automatically.
        
        Args:
            rgb (numpy.ndarray): Input image (could be grayscale or color).

        Returns:
            numpy.ndarray: Image with adjusted contrast.
        """
        alow = rgb.min()
        ahigh = rgb.max()
        amax = 255
        amin = 0
        alpha = ((amax - amin) / (ahigh - alow))
        beta = amin - alow * alpha
        dst = cv2.convertScaleAbs(rgb, alpha=alpha, beta=beta)
        return dst
    
    def process(self, img):
        """
        Apply all preprocessing steps to the input image in sequence:
        1. Histogram equalization (if image is small enough)
        2. Contrast enhancement
        3. Bilateral filtering for edge-preserving smoothing
        4. Denoising using the selected method
        
        Args:
            img (numpy.ndarray): Input image in BGR color space
            
        Returns:
            numpy.ndarray: Preprocessed image with enhanced quality
        """
        # Apply histogram equalization if applicable
        img = self.apply_histogram_equalization(img)
        
        # Enhance contrast using auto contrast adjustment
        img = self.auto_contrast(img)
        
        # Apply bilateral filter for edge-preserving smoothing
        img = cv2.bilateralFilter(img, 
                                self.bilateral_d,
                                self.bilateral_sigma_color,
                                self.bilateral_sigma_space)
        
        # Apply the selected denoising method
        img = self.apply_denoising(img)
        
        return img