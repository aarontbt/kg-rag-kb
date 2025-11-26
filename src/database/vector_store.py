import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from .embeddings import EmbeddingsManager

logger = logging.getLogger(__name__)

class VectorStore:
    """Manage vector database using ChromaDB"""
    
    def __init__(self, 
                 db_path: str = "./vector_db",
                 collection_name: str = "documents",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector store
        
        Args:
            db_path: Path to store ChromaDB data
            collection_name: Name of the collection
            embedding_model: Name of the embedding model
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embeddings_manager = EmbeddingsManager(embedding_model)
        
        self._initialize_client()
        self._get_or_create_collection()
    
    def _initialize_client(self):
        """Initialize ChromaDB client"""
        try:
            # Ensure the database directory exists
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize persistent client
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            logger.info(f"ChromaDB client initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise
    
    def _get_or_create_collection(self):
        """Get or create the collection"""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
            
        except Exception:
            # Create new collection if it doesn't exist
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Document collection for RAG system"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector store
        
        Args:
            documents: List of documents with 'text' and 'metadata' fields
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            logger.warning("No documents to add")
            return False
        
        try:
            texts = [doc['text'] for doc in documents if doc.get('text')]
            metadatas = [doc.get('metadata', {}) for doc in documents if doc.get('text')]
            
            # Generate embeddings
            embeddings = self.embeddings_manager.embed_texts(texts)
            
            if not embeddings:
                logger.error("Failed to generate embeddings")
                return False
            
            # Generate unique IDs
            ids = [f"doc_{i}_{hash(texts[i])}" for i in range(len(texts))]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def search(self, 
               query: str, 
               top_k: int = 5,
               where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Query text
            top_k: Number of results to return
            where: Optional metadata filter
            
        Returns:
            List of search results
        """
        try:
            # Generate query embedding
            query_embedding = self.embeddings_manager.embed_single_text(query)
            
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            return {
                'name': self.collection_name,
                'document_count': count,
                'embedding_dimension': self.embeddings_manager.get_embedding_dimension(),
                'path': str(self.db_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            self.collection = None
            
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
    
    def reset_store(self):
        """Reset the entire vector store"""
        try:
            if self.collection:
                self.delete_collection()
            
            self._get_or_create_collection()
            logger.info("Vector store reset completed")
            
        except Exception as e:
            logger.error(f"Error resetting vector store: {str(e)}")
