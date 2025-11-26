from typing import List, Dict, Any, Optional
import logging
from ..database.vector_store import VectorStore
from ..knowledge_graph.kg_manager import KnowledgeGraphManager
from ..knowledge_graph.entity_extractor import EntityExtractor
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

class DocumentRetriever:
    """Retrieve relevant documents from the vector store"""
    
    def __init__(self, 
                 vector_store: VectorStore,
                 knowledge_graph: KnowledgeGraphManager = None,
                 entity_extractor: EntityExtractor = None):
        """
        Initialize document retriever
        
        Args:
            vector_store: VectorStore instance for document retrieval
            knowledge_graph: KnowledgeGraphManager for graph-based retrieval
            entity_extractor: EntityExtractor for entity-based search
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.entity_extractor = entity_extractor
    
    def retrieve(self, 
                 query: str, 
                 top_k: int = 5,
                 similarity_threshold: float = 0.7,
                 filters: Optional[Dict] = None,
                 use_kg_enhancement: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query text
            top_k: Maximum number of documents to retrieve
            similarity_threshold: Minimum similarity score for documents
            filters: Optional metadata filters
            use_kg_enhancement: Whether to use knowledge graph for enhancement
            
        Returns:
            List of relevant documents with metadata
        """
        try:
            console.print(f"[blue]🔍 Searching: \"{query[:100]}{'...' if len(query) > 100 else ''}\"[/blue]")
            
            # Search in vector store
            vector_results = self.vector_store.search(
                query=query,
                top_k=top_k,  # Get top_k results from vector search
                where=filters
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in vector_results 
                if result['similarity'] >= similarity_threshold
            ]
            
            # Enhance with knowledge graph if available
            kg_enhanced_count = 0
            if use_kg_enhancement and self.knowledge_graph and self.entity_extractor:
                kg_enhanced = self._enhance_with_knowledge_graph(
                    query, filtered_results, top_k
                )
                if kg_enhanced:
                    kg_enhanced_count = len(kg_enhanced)
                    console.print(f"[green]🧠 Knowledge Graph: +{kg_enhanced_count} related documents discovered[/green]")
                    filtered_results.extend(kg_enhanced)
            
            # Remove duplicates and limit to top_k
            seen_docs = set()
            final_results = []
            for result in filtered_results:
                doc_id = result.get('metadata', {}).get('source', '')
                if doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    final_results.append(result)
                    if len(final_results) >= top_k:
                        break
            
            # If no results with threshold, return top results without filtering
            if not final_results and vector_results:
                console.print(f"[yellow]⚠️  Using {min(top_k, len(vector_results))} best matches (below threshold)[/yellow]")
                return vector_results[:top_k]
            
            if kg_enhanced_count > 0:
                console.print(f"[green]✨ Enhanced search with {kg_enhanced_count} knowledge graph connections[/green]")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def _enhance_with_knowledge_graph(self, 
                                    query: str, 
                                    vector_results: List[Dict[str, Any]], 
                                    top_k: int) -> List[Dict[str, Any]]:
        """
        Enhance search results using knowledge graph
        
        Args:
            query: Search query
            vector_results: Results from vector search
            top_k: Maximum number of results to return
            
        Returns:
            List of additional documents from knowledge graph
        """
        try:
            enhanced_results = []
            
            # Extract entities from query
            query_entities = self.entity_extractor.extract_entities(query)
            
            if not query_entities:
                return enhanced_results
            
            console.print(f"[blue]🔍 Discovered {len(query_entities)} entities in query: {', '.join([e['name'] for e in query_entities[:3]])}{'...' if len(query_entities) > 3 else ''}[/blue]")
            
            # For each entity, find related entities and their documents
            for entity in query_entities[:3]:  # Limit to top 3 entities
                entity_name = entity['name']
                
                # Find related entities
                related_entities = self.knowledge_graph.find_related_entities(
                    entity_name, max_depth=2
                )
                
                # Get documents containing these entities
                for rel_entity in related_entities[:5]:  # Limit to top 5 related entities
                    related_docs = self.knowledge_graph.get_entity_documents(
                        rel_entity['entity']
                    )
                    
                    # Get vector store entries for these documents
                    for doc_path in related_docs:
                        # Check if document is already in results
                        already_found = any(
                            result.get('metadata', {}).get('source') == doc_path 
                            for result in vector_results + enhanced_results
                        )
                        
                        if not already_found:
                            # Search for this document in vector store
                            doc_results = self.vector_store.search(
                                query=rel_entity['entity'],  # Use related entity as query
                                top_k=1,
                                where={"source": doc_path}
                            )
                            
                            if doc_results:
                                # Add KG context to the result
                                enhanced_result = doc_results[0].copy()
                                enhanced_result['metadata']['kg_enhanced'] = True
                                enhanced_result['metadata']['kg_entity'] = rel_entity['entity']
                                enhanced_result['metadata']['kg_relationship'] = rel_entity['relationship_info'].get('relationship', 'related')
                                enhanced_result['metadata']['kg_distance'] = rel_entity['distance']
                                
                                # Adjust similarity based on relationship strength
                                boost = 1.0 - (rel_entity['distance'] * 0.2)  # Closer entities get higher boost
                                enhanced_result['similarity'] = min(enhanced_result['similarity'] * boost, 1.0)
                                
                                enhanced_results.append(enhanced_result)
            
            # Sort by similarity and limit
            enhanced_results.sort(key=lambda x: x['similarity'], reverse=True)
            return enhanced_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error enhancing with knowledge graph: {str(e)}")
            return []
    
    def retrieve_by_source(self, 
                          source: str, 
                          top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve documents from a specific source file
        
        Args:
            source: Source file path
            top_k: Maximum number of documents to retrieve
            
        Returns:
            List of documents from the specified source
        """
        try:
            filters = {"source": source}
            
            # Use a simple query to retrieve documents from specific source
            results = self.vector_store.search(
                query="document",  # Simple query to match documents
                top_k=top_k,
                where=filters
            )
            
            logger.info(f"Retrieved {len(results)} documents from source: {source}")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents by source: {str(e)}")
            return []
    
    def retrieve_by_type(self, 
                        file_type: str, 
                        top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve documents of a specific file type
        
        Args:
            file_type: File type (pdf, image, docx, etc.)
            top_k: Maximum number of documents to retrieve
            
        Returns:
            List of documents of the specified type
        """
        try:
            filters = {"file_type": file_type}
            
            # Use a simple query to retrieve documents of specific type
            results = self.vector_store.search(
                query="document",  # Simple query to match documents
                top_k=top_k,
                where=filters
            )
            
            logger.info(f"Retrieved {len(results)} documents of type: {file_type}")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents by type: {str(e)}")
            return []
    
    def get_context_for_query(self, 
                             query: str,
                             max_context_length: int = 4000) -> str:
        """
        Get formatted context for RAG generation
        
        Args:
            query: Query text
            max_context_length: Maximum length of context text
            
        Returns:
            Formatted context string
        """
        try:
            # Retrieve relevant documents
            documents = self.retrieve(query, top_k=5)
            
            if not documents:
                return "No relevant documents found."
            
            # Format context with source information
            context_parts = []
            current_length = 0
            
            for doc in documents:
                source_info = doc.get('metadata', {})
                source_name = source_info.get('source', 'Unknown')
                
                # Format document chunk
                doc_text = f"[Source: {source_name}]\n{doc['text']}"
                
                # Check if adding this document would exceed the limit
                if current_length + len(doc_text) > max_context_length:
                    # Truncate the document to fit
                    remaining_space = max_context_length - current_length - 50  # Leave some buffer
                    if remaining_space > 100:  # Only add if there's meaningful space
                        truncated_text = doc_text[:remaining_space] + "..."
                        context_parts.append(truncated_text)
                    break
                
                context_parts.append(doc_text)
                current_length += len(doc_text)
            
            # Join all context parts
            context = "\n\n---\n\n".join(context_parts)
            
            logger.info(f"Generated context with {len(context_parts)} documents ({len(context)} chars)")
            return context
            
        except Exception as e:
            logger.error(f"Error generating context: {str(e)}")
            return "Error generating context."
