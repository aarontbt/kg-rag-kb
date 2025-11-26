import yaml
import os
import logging
from pathlib import Path

class Config:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.WARNING,  # Only show warnings and errors
            format='%(levelname)s: %(message)s',
            handlers=[logging.NullHandler()]  # Suppress all log output by default
        )
        
        # Suppress specific noisy libraries
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("tokenizers").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("chromadb").setLevel(logging.ERROR)
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            # Return default config if file doesn't exist
            return self._get_default_config()
    
    def _get_default_config(self):
        """Return default configuration"""
        return {
            'database': {
                'path': "./vector_db",
                'collection_name': "documents",
                'embedding_model': "all-MiniLM-L6-v2"
            },
            'documents': {
                'data_path': "./data",
                'supported_formats': ["pdf", "png", "jpg", "jpeg", "pptx", "docx", "txt"],
                'chunk_size': 1000,
                'chunk_overlap': 200
            },
            'ocr': {
                'language': "eng",
                'config': "--psm 6"
            },
            'retrieval': {
                'top_k': 5,
                'similarity_threshold': 0.7
            },
            'llm': {
                'model': "llama2",
                'temperature': 0.7,
                'max_tokens': 1000
            },
            'cli': {
                'max_results_display': 10,
                'show_metadata': True
            }
        }
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation (e.g., 'database.path')"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def ensure_paths(self):
        """Ensure required directories exist"""
        data_path = Path(self.get('documents.data_path'))
        db_path = Path(self.get('database.path'))
        
        data_path.mkdir(exist_ok=True)
        db_path.mkdir(exist_ok=True)
