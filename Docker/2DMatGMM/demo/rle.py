"""
Run-Length Encoding (RLE) implementation for binary masks.
"""
from typing import List, Tuple, Dict, Any
import numpy as np

class RLE:
    """
    Run-Length Encoding class for binary mask compression.
    """
    def __init__(self, data: List[int] = None, shape: Tuple[int, int] = None):
        """
        Initialize RLE object.
        
        Args:
            data (List[int], optional): RLE encoded data. Defaults to None.
            shape (Tuple[int, int], optional): Shape of the original mask. Defaults to None.
        """
        self.data = data
        self.shape = shape
    
    @classmethod
    def from_mask(cls, mask: np.ndarray) -> 'RLE':
        """
        Create RLE object from binary mask.
        
        Args:
            mask (np.ndarray): Binary mask array
            
        Returns:
            RLE: New RLE object
        """
        # Flatten the mask
        flat_mask = mask.ravel()
        
        # Find where values change
        diff = np.diff(flat_mask)
        change_points = np.where(diff != 0)[0] + 1
        
        # Add start and end points
        if flat_mask[0] == 1:
            change_points = np.concatenate(([0], change_points))
        if flat_mask[-1] == 1:
            change_points = np.concatenate((change_points, [len(flat_mask)]))
        
        # Calculate run lengths
        rle_data = []
        for i in range(0, len(change_points), 2):
            if i + 1 < len(change_points):
                start = change_points[i]
                end = change_points[i + 1]
                rle_data.extend([int(start), int(end - start)])
        
        return cls(rle_data, mask.shape)
    
    def to_mask(self) -> np.ndarray:
        """
        Convert RLE data back to binary mask.
        
        Returns:
            np.ndarray: Reconstructed binary mask
        """
        if self.data is None or self.shape is None:
            raise ValueError("RLE data and shape must be set")
            
        mask = np.zeros(self.shape[0] * self.shape[1], dtype=np.uint8)
        
        for i in range(0, len(self.data), 2):
            start = self.data[i]
            length = self.data[i + 1]
            mask[start:start + length] = 1
        
        return mask.reshape(self.shape)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert RLE object to dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary containing RLE data and shape
        """
        return {
            "rle": self.data,
            "shape": list(self.shape) if self.shape else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RLE':
        """
        Create RLE object from dictionary representation.
        
        Args:
            data (Dict[str, Any]): Dictionary containing RLE data and shape
            
        Returns:
            RLE: New RLE object
        """
        return cls(data["rle"], tuple(data["shape"]) if data["shape"] else None)
    
    def __repr__(self) -> str:
        """String representation of RLE object."""
        return f"RLE(data_length={len(self.data) if self.data else 0}, shape={self.shape})" 