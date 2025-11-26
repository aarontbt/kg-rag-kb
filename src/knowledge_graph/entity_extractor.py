import re
import warnings
from typing import List, Dict, Any, Set, Tuple
import logging
from collections import defaultdict
import json
from pathlib import Path

# Try to import spacy, but make it optional
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
except warnings.Warning:
    SPACY_AVAILABLE = False
    spacy = None

logger = logging.getLogger(__name__)

class EntityExtractor:
    """Extract entities from text using multiple approaches"""
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize entity extractor
        
        Args:
            use_spacy: Whether to use spaCy for NER (requires spaCy model)
        """
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.nlp = None
        
        if self.use_spacy:
            try:
                # Try to load spaCy model
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.use_spacy = False
        # Silent initialization - no logging needed
        
        # Define entity patterns for rule-based extraction
        self.patterns = {
            'person': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\b(?:Mr|Mrs|Ms|Dr|Prof)\. [A-Z][a-z]+\b',  # Title + Name
            ],
            'organization': [
                r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*(?:\s+(?:Inc|Corp|LLC|Ltd|Co|Company|Corporation))\b',
                r'\b(?:Google|Microsoft|Apple|Amazon|Facebook|Tesla|OpenAI|Anthropic)\b',
            ],
            'technology': [
                r'\b(?:Python|JavaScript|Java|C\+\+|React|Node\.js|TensorFlow|PyTorch|scikit-learn|pandas|numpy)\b',
                r'\b(?:AI|machine learning|deep learning|neural network|algorithm|database|API|framework)\b',
                r'\b(?:ChromaDB|Neo4j|Docker|Kubernetes|AWS|Azure|GCP|Memgraph|NetworkX)\b',
            ],
            'concept': [
                r'\b(?:artificial intelligence|data science|software development|web development|cloud computing)\b',
                r'\b(?:knowledge graph|vector database|retrieval augmented generation|RAG|natural language processing)\b',
                r'\b(?:computer vision|robotics|automation|blockchain|cybersecurity)\b',
            ],
            'location': [
                r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*(?:\s+(?:City|State|Country|Street|Avenue|Boulevard))?\b',
                r'\b(?:USA|United States|California|New York|San Francisco|Seattle|Boston|Austin)\b',
            ]
        }
        
        # Common words to filter out
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'can', 'shall', 'must', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
    
    def extract_entities(self, text: str, doc_id: str = None) -> List[Dict[str, Any]]:
        """
        Extract entities from text
        
        Args:
            text: Text to extract entities from
            doc_id: Document ID for context
            
        Returns:
            List of extracted entities with metadata
        """
        try:
            entities = []
            
            # Extract using spaCy if available
            if self.use_spacy and self.nlp:
                spacy_entities = self._extract_with_spacy(text, doc_id)
                entities.extend(spacy_entities)
            
            # Extract using rule-based patterns
            rule_entities = self._extract_with_patterns(text, doc_id)
            entities.extend(rule_entities)
            
            # Deduplicate and rank entities
            entities = self._deduplicate_entities(entities)
            
            logger.info(f"Extracted {len(entities)} unique entities from document {doc_id}")
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def _extract_with_spacy(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy NER"""
        try:
            entities = []
            doc = self.nlp(text)
            
            for ent in doc.ents:
                # Filter out very short or very long entities
                if len(ent.text.strip()) < 2 or len(ent.text.strip()) > 50:
                    continue
                
                # Filter out common words
                if ent.text.lower() in self.stopwords:
                    continue
                
                # Map spaCy labels to our types
                entity_type = self._map_spacy_label(ent.label_)
                
                entities.append({
                    'name': ent.text.strip(),
                    'type': entity_type,
                    'confidence': 0.8,  # spaCy NER is generally confident
                    'method': 'spacy',
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'doc_id': doc_id
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error with spaCy extraction: {str(e)}")
            return []
    
    def _map_spacy_label(self, spacy_label: str) -> str:
        """Map spaCy entity labels to our types"""
        mapping = {
            'PERSON': 'person',
            'ORG': 'organization',
            'GPE': 'location',  # Geopolitical Entity
            'LOC': 'location',
            'PRODUCT': 'technology',
            'EVENT': 'concept',
            'WORK_OF_ART': 'concept',
            'LAW': 'concept',
            'LANGUAGE': 'technology',
            'DATE': 'concept',
            'TIME': 'concept',
            'PERCENT': 'concept',
            'MONEY': 'concept',
            'QUANTITY': 'concept',
            'ORDINAL': 'concept',
            'CARDINAL': 'concept'
        }
        
        return mapping.get(spacy_label, 'unknown')
    
    def _extract_with_patterns(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Extract entities using rule-based patterns"""
        try:
            entities = []
            
            for entity_type, patterns in self.patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        entity_text = match.group().strip()
                        
                        # Validate entity
                        if not self._is_valid_entity(entity_text):
                            continue
                        
                        # Calculate confidence based on pattern specificity
                        confidence = self._calculate_pattern_confidence(entity_text, entity_type)
                        
                        entities.append({
                            'name': entity_text,
                            'type': entity_type,
                            'confidence': confidence,
                            'method': 'pattern',
                            'start': match.start(),
                            'end': match.end(),
                            'doc_id': doc_id
                        })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error with pattern extraction: {str(e)}")
            return []
    
    def _is_valid_entity(self, entity_text: str) -> bool:
        """Validate if text is a good entity candidate"""
        # Length checks
        if len(entity_text) < 2 or len(entity_text) > 50:
            return False
        
        # Common words filter
        if entity_text.lower() in self.stopwords:
            return False
        
        # Numeric-only entities
        if entity_text.isdigit():
            return False
        
        # Single character entities
        if len(entity_text.split()) == 1 and len(entity_text) == 1:
            return False
        
        # Common patterns that aren't entities
        if entity_text.startswith(('http', 'www', '@', '#')):
            return False
        
        return True
    
    def _calculate_pattern_confidence(self, entity_text: str, entity_type: str) -> float:
        """Calculate confidence score for pattern-based entities"""
        base_confidence = 0.6
        
        # Boost for well-formed entities
        if entity_type == 'person' and len(entity_text.split()) >= 2:
            base_confidence += 0.2
        elif entity_type == 'organization' and any(suffix in entity_text.lower() 
                                                  for suffix in ['inc', 'corp', 'llc', 'ltd']):
            base_confidence += 0.2
        elif entity_type == 'technology' and entity_text[0].isupper():
            base_confidence += 0.1
        
        # Boost for capitalized entities (except for concepts)
        if entity_text.istitle() and entity_type not in ['concept']:
            base_confidence += 0.1
        
        return min(base_confidence, 0.9)  # Cap at 0.9 to leave room for manual verification
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate entities and choose best extraction"""
        try:
            # Group by normalized entity name
            entity_groups = defaultdict(list)
            
            for entity in entities:
                normalized_name = self._normalize_entity_name(entity['name'])
                entity_groups[normalized_name].append(entity)
            
            # Choose best entity for each group
            deduplicated = []
            
            for normalized_name, group in entity_groups.items():
                # Choose entity with highest confidence
                best_entity = max(group, key=lambda x: x['confidence'])
                
                # Merge methods if multiple extraction methods were used
                methods = [e['method'] for e in group]
                best_entity['method'] = '+'.join(set(methods))
                
                # Boost confidence if multiple methods agree
                if len(set(methods)) > 1:
                    best_entity['confidence'] = min(best_entity['confidence'] + 0.1, 1.0)
                
                deduplicated.append(best_entity)
            
            # Sort by confidence
            deduplicated.sort(key=lambda x: x['confidence'], reverse=True)
            
            return deduplicated
            
        except Exception as e:
            logger.error(f"Error deduplicating entities: {str(e)}")
            return entities
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for deduplication"""
        return name.lower().strip()
    
    def extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities in text
        
        Args:
            text: Text to analyze
            entities: List of entities in the text
            
        Returns:
            List of relationships
        """
        try:
            relationships = []
            entity_names = [e['name'].lower() for e in entities]
            
            # Relationship patterns
            patterns = [
                (r'(\w+(?:\s+\w+)*)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+)*)', 'is_type'),
                (r'(\w+(?:\s+\w+)*)\s+(?:developed|created|built|made)\s+(\w+(?:\s+\w+)*)', 'created'),
                (r'(\w+(?:\s+\w+)*)\s+(?:works?\s+for|is\s+(?:a|an)?\s*part\s+of)\s+(\w+(?:\s+\w+)*)', 'part_of'),
                (r'(\w+(?:\s+\w+)*)\s+(?:uses?|utilizes?|employs)\s+(\w+(?:\s+\w+)*)', 'uses'),
                (r'(\w+(?:\s+\w+)*)\s+(?:owned\s+by|belongs\s+to)\s+(\w+(?:\s+\w+)*)', 'owned_by'),
                (r'(\w+(?:\s+\w+)*)\s+(?:located\s+in|based\s+in)\s+(\w+(?:\s+\w+)*)', 'located_in'),
                (r'(\w+(?:\s+\w+)*)\s+and\s+(\w+(?:\s+\w+)*)', 'related_to')
            ]
            
            # Find relationships
            for pattern, rel_type in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    if len(match.groups()) >= 2:
                        entity1 = match.group(1).lower().strip()
                        entity2 = match.group(2).lower().strip()
                        
                        # Check if both are entities we extracted
                        if (entity1 in entity_names and entity2 in entity_names and 
                            entity1 != entity2):
                            
                            # Get actual entity names (preserve case)
                            actual_entity1 = next((e['name'] for e in entities 
                                                 if e['name'].lower() == entity1), entity1)
                            actual_entity2 = next((e['name'] for e in entities 
                                                 if e['name'].lower() == entity2), entity2)
                            
                            relationships.append({
                                'source': actual_entity1,
                                'target': actual_entity2,
                                'type': rel_type,
                                'confidence': 0.7,  # Base confidence for pattern-based
                                'method': 'pattern',
                                'text': match.group(0)
                            })
            
            return relationships
            
        except Exception as e:
            # Silently handle relationship extraction errors
            return []
    
    def load_custom_patterns(self, patterns_file: str):
        """Load custom entity patterns from a JSON file"""
        try:
            patterns_path = Path(patterns_file)
            if not patterns_path.exists():
                logger.info(f"Custom patterns file {patterns_file} not found, using defaults")
                return
            
            with open(patterns_path, 'r', encoding='utf-8') as f:
                custom_patterns = json.load(f)
            
            # Merge with existing patterns
            for entity_type, type_patterns in custom_patterns.items():
                if entity_type in self.patterns:
                    self.patterns[entity_type].extend(type_patterns)
                else:
                    self.patterns[entity_type] = type_patterns
            
            logger.info(f"Loaded custom patterns from {patterns_file}")
            
        except Exception as e:
            logger.error(f"Error loading custom patterns: {str(e)}")
    
    def get_entity_types(self) -> List[str]:
        """Get list of entity types the extractor can identify"""
        return list(self.patterns.keys())
