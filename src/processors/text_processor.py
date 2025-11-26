from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """Process plain text files"""
    
    def __init__(self):
        pass
    
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a text file and return extracted text with metadata
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of dictionaries containing extracted text and metadata
        """
        file_path = Path(file_path)
        
        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            if text and text.strip():
                chunks = [{
                    'text': text.strip(),
                    'metadata': {
                        'source': str(file_path),
                        'file_type': 'txt',
                        'size': len(text),
                        'encoding': 'utf-8'
                    }
                }]
                
                logger.info(f"Successfully processed text file: {file_path}")
                return chunks
            else:
                logger.warning(f"Empty text file: {file_path}")
                return []
                
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                
                if text and text.strip():
                    chunks = [{
                        'text': text.strip(),
                        'metadata': {
                            'source': str(file_path),
                            'file_type': 'txt',
                            'size': len(text),
                            'encoding': 'latin-1'
                        }
                    }]
                    
                    logger.info(f"Successfully processed text file: {file_path}")
                    return chunks
                    
            except Exception as e:
                logger.error(f"Error processing text file {file_path}: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            return []
    
    def is_supported(self, file_path: str) -> bool:
        """Check if the file is a supported text format"""
        return Path(file_path).suffix.lower() == '.txt'
