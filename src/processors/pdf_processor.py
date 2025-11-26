import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDF documents to extract text and metadata"""
    
    def __init__(self):
        pass
    
    def process(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a PDF file and return extracted text chunks with metadata
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        file_path = Path(file_path)
        chunks = []
        
        try:
            # Try pdfplumber first for better text extraction
            chunks.extend(self._extract_with_pdfplumber(file_path))
            
            # Fallback to PyMuPDF if pdfplumber fails
            if not chunks:
                chunks.extend(self._extract_with_pymupdf(file_path))
                
            logger.info(f"Successfully processed PDF: {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return []
    
    def _extract_with_pdfplumber(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract text using pdfplumber"""
        chunks = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        chunks.append({
                            'text': text.strip(),
                            'metadata': {
                                'source': str(file_path),
                                'file_type': 'pdf',
                                'page': page_num + 1,
                                'total_pages': len(pdf.pages)
                            }
                        })
        except Exception as e:
            logger.warning(f"pdfplumber failed for {file_path}: {str(e)}")
            
        return chunks
    
    def _extract_with_pymupdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract text using PyMuPDF as fallback"""
        chunks = []
        
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text()
                
                if text and text.strip():
                    chunks.append({
                        'text': text.strip(),
                        'metadata': {
                            'source': str(file_path),
                            'file_type': 'pdf',
                            'page': page_num + 1,
                            'total_pages': total_pages
                        }
                    })
            
            doc.close()
            
        except Exception as e:
            logger.error(f"PyMuPDF failed for {file_path}: {str(e)}")
            
        return chunks
    
    def is_supported(self, file_path: str) -> bool:
        """Check if the file is a supported PDF format"""
        return Path(file_path).suffix.lower() == '.pdf'
