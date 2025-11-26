import networkx as nx
import json
import re
from typing import Dict, List, Set, Tuple, Any, Optional
from pathlib import Path
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    """Manage knowledge graph using NetworkX"""
    
    def __init__(self, storage_path: str = "./knowledge_graph.json"):
        """
        Initialize knowledge graph manager
        
        Args:
            storage_path: Path to save/load the graph
        """
        self.storage_path = Path(storage_path)
        self.graph = nx.DiGraph()  # Directed graph for relationships
        self.entity_index = defaultdict(list)  # Map entities to document IDs
        
    def add_document_entities(self, doc_id: str, text: str, entities: List[Dict[str, Any]]):
        """
        Add entities and relationships from a document to the knowledge graph
        
        Args:
            doc_id: Document identifier
            text: Document text
            entities: List of extracted entities with metadata
        """
        try:
            # Create document node
            self.graph.add_node(doc_id, 
                              type="document",
                              text=text[:500] + "..." if len(text) > 500 else text)
            
            # Add entity nodes and relationships
            for entity in entities:
                entity_name = entity.get('name', '').strip()
                entity_type = entity.get('type', 'unknown')
                
                if not entity_name:
                    continue
                
                # Create entity node if it doesn't exist
                if entity_name not in self.graph:
                    self.graph.add_node(entity_name, 
                                      type=entity_type,
                                      documents=set())
                
                # Add document to entity's documents set
                if hasattr(self.graph.nodes[entity_name], 'documents'):
                    self.graph.nodes[entity_name]['documents'].add(doc_id)
                else:
                    self.graph.nodes[entity_name]['documents'] = {doc_id}
                
                # Connect document to entity
                self.graph.add_edge(doc_id, entity_name, relationship="contains")
                
                # Track entity in index
                self.entity_index[entity_name.lower()].append(doc_id)
            
            # Extract and add relationships between entities
            self._extract_relationships(doc_id, text, entities)
            
            logger.info(f"Added {len(entities)} entities from document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error adding entities to graph: {str(e)}")
    
    def _extract_relationships(self, doc_id: str, text: str, entities: List[Dict[str, Any]]):
        """Extract relationships between entities in the text"""
        try:
            # Simple pattern-based relationship extraction
            relationship_patterns = [
                (r'(\w+)\s+(is|are|was|were)\s+(a|an|the)?\s*(\w+)', "is_type"),
                (r'(\w+)\s+(works?\s+for|is\s+(a|an)?\s*part\s+of)\s+(\w+)', "part_of"),
                (r'(\w+)\s+(developed|created|invented)\s+(\w+)', "created"),
                (r'(\w+)\s+(uses?\s+)?(based\s+on)?\s+(\w+)', "uses"),
                (r'(\w+)\s+and\s+(\w+)', "related_to")
            ]
            
            entity_names = [e.get('name', '').strip().lower() for e in entities if e.get('name')]
            
            for pattern, rel_type in relationship_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    # Get the captured groups
                    groups = match.groups()
                    
                    for i in range(len(groups) - 1):
                        entity1 = groups[i].lower()
                        entity2 = groups[-1].lower()  # Last group is usually the second entity
                        
                        # Check if these are valid entities from our list
                        if (entity1 in entity_names and entity2 in entity_names and 
                            entity1 != entity2):
                            
                            # Get the actual entity names (preserve case)
                            actual_entity1 = next((e for e in entity_names if e == entity1), entity1)
                            actual_entity2 = next((e for e in entity_names if e == entity2), entity2)
                            
                            # Add relationship if nodes exist
                            if (actual_entity1 in self.graph.nodes and 
                                actual_entity2 in self.graph.nodes):
                                
                                # Add edge with relationship type and document
                                edge_key = (actual_entity1, actual_entity2)
                                
                                if self.graph.has_edge(*edge_key):
                                    # Update existing edge
                                    self.graph.edges[edge_key]['documents'].add(doc_id)
                                    if rel_type not in self.graph.edges[edge_key]['types']:
                                        self.graph.edges[edge_key]['types'].append(rel_type)
                                else:
                                    # Create new edge
                                    self.graph.add_edge(actual_entity1, actual_entity2,
                                                     relationship=rel_type,
                                                     types=[rel_type],
                                                     documents={doc_id})
            
        except Exception as e:
            logger.error(f"Error extracting relationships: {str(e)}")
    
    def find_related_entities(self, entity_name: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Find entities related to a given entity
        
        Args:
            entity_name: Name of the entity to search for
            max_depth: Maximum depth of relationship traversal
            
        Returns:
            List of related entities with relationship info
        """
        try:
            entity_name_lower = entity_name.lower()
            
            # Find exact or similar entity names
            matching_entities = []
            for node in self.graph.nodes():
                if (node.lower() == entity_name_lower or 
                    entity_name_lower in node.lower() or 
                    node.lower() in entity_name_lower):
                    matching_entities.append(node)
            
            if not matching_entities:
                return []
            
            # Explore relationships
            related_entities = []
            visited = set()
            
            for entity in matching_entities:
                self._explore_relationships(entity, related_entities, visited, max_depth, 0)
            
            # Remove the original entity from results
            related_entities = [r for r in related_entities if r['entity'] not in matching_entities]
            
            return related_entities
            
        except Exception as e:
            logger.error(f"Error finding related entities: {str(e)}")
            return []
    
    def _explore_relationships(self, 
                             current_entity: str, 
                             results: List[Dict[str, Any]], 
                             visited: Set[str], 
                             max_depth: int, 
                             current_depth: int):
        """Recursively explore relationships"""
        if current_depth >= max_depth or current_entity in visited:
            return
        
        visited.add(current_entity)
        
        # Get all neighbors (both incoming and outgoing)
        neighbors = list(self.graph.predecessors(current_entity)) + list(self.graph.successors(current_entity))
        
        for neighbor in neighbors:
            if neighbor not in visited:
                # Get relationship info
                edge_data = self.graph.get_edge_data(current_entity, neighbor)
                reverse_edge_data = self.graph.get_edge_data(neighbor, current_entity)
                
                relationship_info = {}
                
                if edge_data:
                    relationship_info.update({
                        'relationship': edge_data.get('relationship', 'related'),
                        'direction': 'outgoing',
                        'types': edge_data.get('types', []),
                        'documents': edge_data.get('documents', set())
                    })
                elif reverse_edge_data:
                    relationship_info.update({
                        'relationship': reverse_edge_data.get('relationship', 'related'),
                        'direction': 'incoming',
                        'types': reverse_edge_data.get('types', []),
                        'documents': reverse_edge_data.get('documents', set())
                    })
                
                # Add to results
                results.append({
                    'entity': neighbor,
                    'entity_type': self.graph.nodes[neighbor].get('type', 'unknown'),
                    'distance': current_depth + 1,
                    'relationship_info': relationship_info
                })
                
                # Continue exploring
                self._explore_relationships(neighbor, results, visited, max_depth, current_depth + 1)
    
    def get_entity_documents(self, entity_name: str) -> List[str]:
        """Get all documents that contain a specific entity"""
        try:
            entity_name_lower = entity_name.lower()
            
            # Find matching entity
            for node in self.graph.nodes():
                if node.lower() == entity_name_lower:
                    return list(self.graph.nodes[node].get('documents', set()))
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting entity documents: {str(e)}")
            return []
    
    def search_entities(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for entities matching the query
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching entities
        """
        try:
            query_lower = query.lower()
            results = []
            
            for node in self.graph.nodes():
                if self.graph.nodes[node].get('type') == 'document':
                    continue  # Skip document nodes
                
                # Check if entity name matches
                if (query_lower in node.lower() or 
                    node.lower() in query_lower):
                    
                    # Get entity info
                    entity_info = self.graph.nodes[node]
                    results.append({
                        'entity': node,
                        'type': entity_info.get('type', 'unknown'),
                        'document_count': len(entity_info.get('documents', set())),
                        'relevance': self._calculate_relevance(node, query)
                    })
            
            # Sort by relevance and limit
            results.sort(key=lambda x: x['relevance'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Error searching entities: {str(e)}")
            return []
    
    def _calculate_relevance(self, entity: str, query: str) -> float:
        """Calculate relevance score for entity-query match"""
        entity_lower = entity.lower()
        query_lower = query.lower()
        
        # Exact match
        if entity_lower == query_lower:
            return 1.0
        
        # Query contains entity
        if query_lower in entity_lower:
            return 0.8
        
        # Entity contains query
        if entity_lower in query_lower:
            return 0.6
        
        # Partial match
        if any(word in entity_lower for word in query_lower.split()):
            return 0.4
        
        return 0.0
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        try:
            entity_nodes = [n for n in self.graph.nodes() 
                          if self.graph.nodes[n].get('type') != 'document']
            document_nodes = [n for n in self.graph.nodes() 
                            if self.graph.nodes[n].get('type') == 'document']
            
            # Entity type distribution
            entity_types = defaultdict(int)
            for entity in entity_nodes:
                entity_type = self.graph.nodes[entity].get('type', 'unknown')
                entity_types[entity_type] += 1
            
            return {
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'entity_count': len(entity_nodes),
                'document_count': len(document_nodes),
                'entity_types': dict(entity_types),
                'avg_connections': self.graph.number_of_edges() / max(len(entity_nodes), 1)
            }
            
        except Exception as e:
            logger.error(f"Error getting graph stats: {str(e)}")
            return {}
    
    def save_graph(self):
        """Save the knowledge graph to disk"""
        try:
            # Convert NetworkX graph to serializable format
            data = {
                'nodes': [],
                'edges': []
            }
            
            # Serialize nodes
            for node, attrs in self.graph.nodes(data=True):
                # Convert sets to lists for JSON serialization
                serializable_attrs = {}
                for k, v in attrs.items():
                    if isinstance(v, set):
                        serializable_attrs[k] = list(v)
                    else:
                        serializable_attrs[k] = v
                
                data['nodes'].append({
                    'id': node,
                    'attributes': serializable_attrs
                })
            
            # Serialize edges
            for source, target, attrs in self.graph.edges(data=True):
                # Convert sets to lists for JSON serialization
                serializable_attrs = {}
                for k, v in attrs.items():
                    if isinstance(v, set):
                        serializable_attrs[k] = list(v)
                    else:
                        serializable_attrs[k] = v
                
                data['edges'].append({
                    'source': source,
                    'target': target,
                    'attributes': serializable_attrs
                })
            
            # Save to file
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Knowledge graph saved to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Error saving graph: {str(e)}")
    
    def load_graph(self):
        """Load the knowledge graph from disk"""
        try:
            if not self.storage_path.exists():
                logger.info("No existing knowledge graph found, starting fresh")
                return
            
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Recreate graph from data
            self.graph.clear()
            
            # Add nodes
            for node_data in data.get('nodes', []):
                node_id = node_data['id']
                attrs = node_data.get('attributes', {})
                
                # Convert lists back to sets
                for k, v in attrs.items():
                    if isinstance(v, list) and k in ['documents']:
                        attrs[k] = set(v)
                
                self.graph.add_node(node_id, **attrs)
            
            # Add edges
            for edge_data in data.get('edges', []):
                source = edge_data['source']
                target = edge_data['target']
                attrs = edge_data.get('attributes', {})
                
                # Convert lists back to sets
                for k, v in attrs.items():
                    if isinstance(v, list) and k in ['documents']:
                        attrs[k] = set(v)
                
                self.graph.add_edge(source, target, **attrs)
            
            logger.info(f"Knowledge graph loaded from {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Error loading graph: {str(e)}")
    
    def visualize_subgraph(self, entity_name: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        Create a visualization-ready subgraph around an entity
        
        Args:
            entity_name: Central entity
            max_depth: Maximum depth from the central entity
            
        Returns:
            Dictionary with nodes and edges for visualization
        """
        try:
            # Find related entities
            related_entities = self.find_related_entities(entity_name, max_depth)
            
            # Build subgraph
            nodes = []
            edges = []
            visited = set([entity_name])
            
            # Add central node
            nodes.append({
                'id': entity_name,
                'label': entity_name,
                'type': self.graph.nodes[entity_name].get('type', 'unknown'),
                'size': 20,
                'color': '#ff6b6b'
            })
            
            # Add related entities and connections
            for rel_info in related_entities:
                related_entity = rel_info['entity']
                
                if related_entity not in visited:
                    visited.add(related_entity)
                    
                    nodes.append({
                        'id': related_entity,
                        'label': related_entity,
                        'type': rel_info['entity_type'],
                        'size': 10 + (10 - rel_info['distance'] * 2),
                        'color': self._get_node_color(rel_info['entity_type'])
                    })
                    
                    # Add edge
                    edges.append({
                        'from': entity_name,
                        'to': related_entity,
                        'label': rel_info['relationship_info'].get('relationship', 'related'),
                        'width': max(1, 5 - rel_info['distance'])
                    })
            
            return {
                'nodes': nodes,
                'edges': edges,
                'stats': {
                    'total_nodes': len(nodes),
                    'total_edges': len(edges),
                    'max_depth': max_depth
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating subgraph visualization: {str(e)}")
            return {'nodes': [], 'edges': []}
    
    def _get_node_color(self, entity_type: str) -> str:
        """Get color for entity type"""
        color_map = {
            'person': '#4ecdc4',
            'organization': '#45b7d1',
            'location': '#96ceb4',
            'technology': '#ffeaa7',
            'concept': '#dfe6e9',
            'unknown': '#b2bec3'
        }
        return color_map.get(entity_type.lower(), '#b2bec3')
