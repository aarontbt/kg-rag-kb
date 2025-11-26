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
        """Extract text and tables using pdfplumber"""
        chunks = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract tables first
                    tables = page.extract_tables()

                    # Extract text
                    text = page.extract_text()

                    # Add text chunk
                    if text and text.strip():
                        chunks.append({
                            'text': text.strip(),
                            'metadata': {
                                'source': str(file_path),
                                'file_type': 'pdf',
                                'page': page_num + 1,
                                'content_type': 'text',
                                'total_pages': len(pdf.pages)
                            }
                        })

                    # Add table chunks
                    if tables:
                        for i, table in enumerate(tables):
                            if table and len(table) > 1:  # At least header + 1 row
                                chunks.append({
                                    'text': self._table_to_markdown(table),
                                    'metadata': {
                                        'source': str(file_path),
                                        'file_type': 'pdf',
                                        'page': page_num + 1,
                                        'content_type': 'table',
                                        'table_number': i + 1,
                                        'total_pages': len(pdf.pages),
                                        'table_metadata': {
                                            'num_rows': len(table),
                                            'num_columns': len(table[0]) if table else 0,
                                            'headers': table[0] if table else [],
                                            'extraction_method': 'pdfplumber'
                                        },
                                        'structured_data': {
                                            'format': 'json',
                                            'data': self._table_to_dict(table)
                                        }
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
                            'content_type': 'text',
                            'total_pages': total_pages
                        }
                    })

            doc.close()

        except Exception as e:
            logger.error(f"PyMuPDF failed for {file_path}: {str(e)}")

        return chunks
    
    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """
        Convert table to markdown format for embedding

        Args:
            table: List of lists representing table rows

        Returns:
            Markdown-formatted table string
        """
        if not table:
            return ""

        lines = []

        # Header row
        lines.append('| ' + ' | '.join(str(cell or '').strip() for cell in table[0]) + ' |')

        # Separator
        lines.append('| ' + ' | '.join(['---'] * len(table[0])) + ' |')

        # Data rows
        for row in table[1:]:
            lines.append('| ' + ' | '.join(str(cell or '').strip() for cell in row) + ' |')

        return '\n'.join(lines)

    def _table_to_dict(self, table: List[List[str]]) -> List[Dict]:
        """
        Convert table to list of dictionaries for structured storage

        Args:
            table: List of lists representing table rows

        Returns:
            List of dictionaries with header keys
        """
        if not table or len(table) < 2:
            return []

        headers = [str(cell or '').strip() for cell in table[0]]
        return [
            {headers[i]: str(row[i] or '').strip() for i in range(min(len(headers), len(row)))}
            for row in table[1:]
        ]

    def is_supported(self, file_path: str) -> bool:
        """Check if the file is a supported PDF format"""
        return Path(file_path).suffix.lower() == '.pdf'
