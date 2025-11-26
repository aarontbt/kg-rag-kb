import pytesseract
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
import logging
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    numpy = None

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Process image files to extract text using OCR"""
    
    def __init__(self, language='eng', config='--psm 6'):
        self.language = language
        self.config = config
    
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process an image file and return extracted text with metadata
        
        Args:
            file_path: Path to the image file
            
        Returns:
            List of dictionaries containing extracted text and metadata
        """
        file_path = Path(file_path)
        
        try:
            # Preprocess image for better OCR
            processed_image = self._preprocess_image(file_path)
            
            # Extract text using OCR
            text = pytesseract.image_to_string(
                processed_image, 
                lang=self.language, 
                config=self.config
            )
            
            if text and text.strip():
                chunks = [{
                    'text': text.strip(),
                    'metadata': {
                        'source': str(file_path),
                        'file_type': 'image',
                        'format': file_path.suffix.lower(),
                        'size': self._get_image_size(file_path)
                    }
                }]
                
                logger.info(f"Successfully processed image: {file_path}")
                return chunks
            else:
                logger.warning(f"No text found in image: {file_path}")
                return []
                
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {str(e)}")
            return []
    
    def _preprocess_image(self, file_path: Path) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            if not CV2_AVAILABLE:
                logger.warning("OpenCV not available, skipping image preprocessing")
                return Image.open(file_path)
                
            # Open image with PIL
            image = Image.open(file_path)
            
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Convert to numpy array for OpenCV operations
            img_array = numpy.array(image)
            
            # Apply thresholding
            _, img_array = cv2.threshold(
                img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            # Convert back to PIL Image
            return Image.fromarray(img_array)
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed, using original: {str(e)}")
            return Image.open(file_path)
    
    def _get_image_size(self, file_path: Path) -> tuple:
        """Get image dimensions"""
        try:
            with Image.open(file_path) as img:
                return img.size  # (width, height)
        except:
            return (0, 0)
    
    def is_supported(self, file_path: str) -> bool:
        """Check if the file is a supported image format"""
        supported_formats = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
        return Path(file_path).suffix.lower() in supported_formats
