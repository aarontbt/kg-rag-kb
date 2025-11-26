# PDF Tabular Data Processing - Implementation Specification

## Document Overview

**Purpose**: Comprehensive analysis and implementation plan for improving PDF table ingestion and storage in the KG-RAG system.

**Status**: Phase 1 implemented, Phases 2-4 planned for future development.

**Last Updated**: 2025-11-26

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Research Findings](#research-findings)
3. [Implementation Phases](#implementation-phases)
4. [Technical Specifications](#technical-specifications)
5. [References](#references)

---

## Current State Analysis

### PDF Processing Implementation

**File**: `src/processors/pdf_processor.py`

**Current Behavior**:
- Uses **pdfplumber** as primary extraction method (lines 43-64)
- Falls back to **PyMuPDF/fitz** if pdfplumber fails (lines 66-94)
- Extracts text **page-by-page** only
- **NO table detection or extraction** in current implementation

**Key Limitations**:
1. ❌ Tables extracted as unstructured plain text
2. ❌ No cell-level data preservation
3. ❌ No table structure metadata
4. ❌ No distinction between text content and tabular content
5. ❌ Table structure and relationships lost in conversion

### Existing Table Support (DOCX/PPTX)

**File**: `src/processors/docx_processor.py` (lines 127-167)

The DOCX processor DOES extract tables using pipe-delimited format:

```python
def _extract_single_table_text(self, table) -> str:
    for row in table.rows:
        cells_text = []
        for cell in row.cells:
            cells_text.append(cell.text.strip())
        rows_text.append(' | '.join(cells_text))  # Pipe-delimited format
    return '\n'.join(rows_text)
```

**Issues with Current Approach**:
- Converts structured data → unstructured text
- Loses column/row relationships
- Cannot query individual cells
- Header rows not distinguished from data rows
- No schema preservation

### Storage Architecture

**Vector Store**: `src/database/vector_store.py`
- Uses **ChromaDB** for vector storage
- Embedding model: `all-MiniLM-L6-v2` (384 dimensions)
- Metadata structure is flat dictionary only

**Current Metadata Schema**:
```python
{
    'source': str,        # File path
    'file_type': 'pdf',
    'page': int,
    'total_pages': int
}
```

**Storage Locations**:
- Vector Database: `./vector_db/` (ChromaDB)
- Knowledge Graph: `./knowledge_graph.json`
- Raw Documents: `./data/`
- Configuration: `./config.yaml`

---

## Research Findings

### Best Python Libraries for PDF Table Extraction (2025)

Based on comprehensive industry research:

#### 1. **pdfplumber** ✅ (Already installed)
- **Strengths**:
  - Versatile, accurate text and table extraction
  - Built-in `extract_tables()` method
  - Good with complex table structures
  - Fast (~0.10 seconds per page)
- **Best for**: General-purpose PDF processing
- **Output**: Lists of lists (rows/cells)

#### 2. **Camelot**
- **Strengths**:
  - Specialized for table extraction
  - Two modes: `lattice` (bordered) and `stream` (spacing-based)
  - High accuracy for structured tables
  - Outputs to Pandas DataFrames, CSV, JSON, HTML
- **Limitations**: Text-based PDFs only (no scanned documents)
- **Best for**: Production systems requiring high accuracy

#### 3. **Tabula-py**
- **Strengths**:
  - Industry standard
  - Java-based with Python wrapper
  - Good for complex multi-page tables
- **Best for**: Enterprise applications

#### 4. **PyMuPDF4LLM**
- **Strengths**:
  - Clean markdown output
  - Table formatting preserved
  - Good integration with LLM workflows
- **Best for**: RAG pipelines

### Best Practices for RAG with Structured Data

Key findings from research on RAG systems with tabular data:

#### Schema and Metadata Management
- Extract and store metadata to create a knowledge graph of the schema
- Store 3-5 representative rows per table for context
- Include column headers, data types, table dimensions

#### Data Structure Consistency
- Maintain consistent schema for reliable retrieval
- Use JSON/YAML for readability and flexibility
- Use Avro/Parquet for large data volumes

#### Knowledge Graph Integration
- GraphRAG addresses traditional RAG limitations
- Incorporate structured domain knowledge
- Leverage semantic relationships in knowledge graphs
- Enhances retrieval accuracy and supports reasoning

#### Hybrid Approaches
- Combine knowledge graphs with native vector search
- Use structured query tools for precise questions
- Use vector search for semantic queries
- Balance keyword and embedding-based retrieval

#### Security and Accuracy
- Implement robust permission checks (row-level, role-based)
- Sandbox query generation to prevent malicious use
- Emphasize accuracy, correctness, and security

### Storage Approach Comparison

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Pipe-delimited text** (current) | Simple, works with any vector DB | Loses structure, hard to query | Basic RAG, no table queries |
| **Markdown tables** | Human-readable, LLM-friendly | Still loses structure | RAG with LLM consumption |
| **JSON in metadata** | Preserves structure, queryable | Size limits in metadata | Small tables (<100 cells) |
| **Dual storage** (vector + structured) | Best of both worlds | More complex | Production systems |
| **Parquet files** | Efficient for large tables | Extra infrastructure | Big data, analytics |

---

## Implementation Phases

### ✅ Phase 1: Enhanced PDF Table Extraction (IMPLEMENTED)

**Timeline**: 1-2 days

**Objectives**:
1. Enable pdfplumber's `extract_tables()` in pdf_processor.py
2. Add `content_type` field to metadata ('table' vs 'text')
3. Store table headers in metadata
4. Create separate chunks for tables vs text
5. Convert tables to markdown format for better LLM consumption

**Changes**:
- Modified `src/processors/pdf_processor.py`
- Added methods: `_table_to_markdown()`, `_table_to_dict()`
- Enhanced metadata structure with table-specific fields

**Expected Benefits**:
- Tables preserved as structured data
- Better retrieval of tabular content
- LLM-friendly markdown format
- Foundation for future enhancements

### 📋 Phase 2: Enhanced Storage Schema (PLANNED)

**Timeline**: 1 week

**Objectives**:
1. Add table processing configuration to config.yaml
2. Implement table-aware chunking (respect table boundaries)
3. Add Camelot library for improved detection
4. Enhance DOCX/PPTX processors with same structure
5. Add extraction method comparison/fallback logic

**Configuration Changes**:
```yaml
documents:
  # NEW: Table processing settings
  tables:
    extract_tables: true
    preserve_structure: true
    extraction_method: "pdfplumber"  # or "camelot", "tabula"
    confidence_threshold: 0.7
    max_table_size_mb: 10

  # Chunking strategy
  chunking:
    enabled: true
    chunk_size: 1000
    chunk_overlap: 200
    respect_table_boundaries: true  # Don't split tables across chunks
```

**Metadata Enhancements**:
```python
{
    'source': str,
    'file_type': str,
    'page': int,
    'content_type': 'table' | 'text' | 'mixed',

    'table_metadata': {
        'table_number': int,
        'num_rows': int,
        'num_columns': int,
        'headers': List[str],
        'column_types': List[str],
        'extraction_method': str,
        'confidence_score': float,
    },

    'structured_data': {
        'format': 'json' | 'csv' | 'parquet',
        'data': Any,
        'schema': Dict
    },

    'surrounding_text': {
        'before': str,
        'after': str,
        'caption': str
    }
}
```

**Dependencies**:
- Add Camelot: `pip install camelot-py[cv]`
- May require system dependencies (Ghostscript, Tkinter)

### 📋 Phase 3: Dual Storage Strategy (PLANNED)

**Timeline**: 2-3 weeks

**Objectives**:
1. Implement separate storage for large tables
2. Add Parquet/SQLite backend for structured queries
3. Create lazy loading mechanism
4. Add table validation and quality scoring
5. Implement OCR support for scanned PDF tables

**Storage Architecture**:
```
./data/                    # Input PDFs
./vector_db/              # ChromaDB (embeddings + metadata)
./table_store/            # Structured table storage
  ├── tables.db          # SQLite for queryable tables
  ├── tables/            # Parquet files for large tables
  │   ├── doc1_table1.parquet
  │   └── doc2_table1.parquet
  └── schemas/           # JSON schemas for each table
```

**Implementation Details**:

1. **Vector Store Changes** (`src/database/vector_store.py`):
   - Add table reference handling
   - Implement external storage for large tables
   - Add lazy loading for structured data

2. **New Table Store** (`src/database/table_store.py`):
   ```python
   class TableStore:
       def __init__(self, storage_path: str = "./table_store"):
           self.storage_path = Path(storage_path)
           self.db_path = self.storage_path / "tables.db"
           self.parquet_path = self.storage_path / "tables"

       def store_table(self, table_id: str, data: pd.DataFrame,
                      metadata: Dict) -> str:
           """Store table and return reference ID"""

       def retrieve_table(self, table_id: str) -> pd.DataFrame:
           """Retrieve table by ID"""

       def query_table(self, table_id: str, query: str) -> pd.DataFrame:
           """Execute SQL query on table"""
   ```

3. **OCR Integration**:
   - Use pytesseract for scanned PDF tables
   - Add confidence scoring for OCR results
   - Implement manual review workflow for low-confidence extractions

**Dependencies**:
- Add Parquet support: `pip install pyarrow`
- SQLite (included in Python standard library)
- Enhanced pytesseract configuration

### 📋 Phase 4: Knowledge Graph Enhancement (PLANNED)

**Timeline**: 1-2 weeks

**Objectives**:
1. Add table entities to knowledge graph
2. Create relationships between tables and entities
3. Enable table-aware graph queries
4. Implement table-entity linking
5. Add table provenance tracking

**Knowledge Graph Changes** (`src/knowledge_graph/kg_manager.py`):

```python
def add_table_entities(self, doc_id: str, table_data: Dict):
    """
    Add table as structured node in knowledge graph

    Creates:
    - TABLE node with schema metadata
    - COLUMN nodes for each column
    - Links to entities mentioned in cells
    - Relationships extracted from table data
    """
    table_node_id = f"{doc_id}_table_{table_data['table_number']}"

    # Add table node
    self.graph.add_node(
        table_node_id,
        node_type='table',
        headers=table_data['headers'],
        num_rows=table_data['num_rows'],
        num_columns=table_data['num_columns']
    )

    # Link document to table
    self.graph.add_edge(
        doc_id, table_node_id,
        relationship='contains_table'
    )

    # Add column nodes
    for col_idx, header in enumerate(table_data['headers']):
        col_node_id = f"{table_node_id}_col_{col_idx}"
        self.graph.add_node(
            col_node_id,
            node_type='column',
            name=header,
            data_type=table_data['column_types'][col_idx]
        )
        self.graph.add_edge(
            table_node_id, col_node_id,
            relationship='has_column'
        )

    # Extract and link entities from cells
    self._extract_cell_entities(table_node_id, table_data)

    # Extract relationships from table structure
    self._extract_table_relationships(table_node_id, table_data)

def _extract_cell_entities(self, table_id: str, table_data: Dict):
    """Extract entities mentioned in table cells"""
    for row_idx, row in enumerate(table_data['data']):
        for col_idx, cell in enumerate(row):
            entities = self.entity_extractor.extract_entities(str(cell))
            for entity in entities:
                # Link entity to table
                self.graph.add_edge(
                    table_id, entity['id'],
                    relationship='mentions',
                    row=row_idx,
                    column=col_idx
                )

def _extract_table_relationships(self, table_id: str, table_data: Dict):
    """Extract relationships from table structure"""
    # Example: If table has "Person" and "Organization" columns,
    # create "works_for" relationships
    # This requires domain-specific logic
    pass
```

**Graph Query Enhancements**:
```python
# New query capabilities
def find_tables_by_entity(self, entity_id: str) -> List[Dict]:
    """Find all tables that mention an entity"""

def get_table_context(self, table_id: str) -> Dict:
    """Get surrounding context and related entities for a table"""

def query_related_tables(self, table_id: str) -> List[str]:
    """Find tables related through shared entities"""
```

---

## Technical Specifications

### Phase 1 Implementation Details

#### Modified Methods

**`_extract_with_pdfplumber()`**:
```python
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
                for i, table in enumerate(tables):
                    if table and len(table) > 1:
                        chunks.append({
                            'text': self._table_to_markdown(table),
                            'metadata': {
                                'source': str(file_path),
                                'file_type': 'pdf',
                                'page': page_num + 1,
                                'content_type': 'table',
                                'table_number': i + 1,
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
```

#### New Helper Methods

**`_table_to_markdown()`**:
```python
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
    lines.append('| ' + ' | '.join(str(cell or '') for cell in table[0]) + ' |')

    # Separator
    lines.append('| ' + ' | '.join(['---'] * len(table[0])) + ' |')

    # Data rows
    for row in table[1:]:
        lines.append('| ' + ' | '.join(str(cell or '') for cell in row) + ' |')

    return '\n'.join(lines)
```

**`_table_to_dict()`**:
```python
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

    headers = table[0]
    return [
        {headers[i]: row[i] for i in range(len(headers))}
        for row in table[1:]
    ]
```

### Testing Considerations

**Test Cases for Phase 1**:
1. PDF with no tables → Should extract text only
2. PDF with single table → Should create separate chunks
3. PDF with multiple tables per page → Each table gets own chunk
4. PDF with tables and text → Both extracted separately
5. Empty/malformed tables → Handled gracefully

**Test Files Needed**:
- Simple table (2x2)
- Complex table (many rows/columns)
- Table with merged cells
- Table with empty cells
- Multiple tables per page

### Performance Considerations

**Expected Impact**:
- Slightly slower processing due to table extraction
- Additional memory for table structures
- More chunks generated (separate table chunks)

**Optimizations**:
- Skip empty tables
- Set max table size limit
- Cache table extraction results
- Batch process multiple pages

---

## Future Enhancements

### Beyond Phase 4

1. **Advanced Table Understanding**:
   - Table type classification (data, summary, comparison)
   - Automatic schema inference
   - Data type detection (numeric, date, categorical)

2. **Table Querying Interface**:
   - Natural language to SQL translation
   - Table-specific retrieval ranking
   - Cross-table joins and aggregations

3. **Multi-page Table Support**:
   - Detect tables spanning multiple pages
   - Merge continued tables
   - Preserve context across pages

4. **Table Quality Metrics**:
   - Extraction confidence scoring
   - Data completeness metrics
   - Schema validation

5. **Visualization Support**:
   - Generate table visualizations
   - Chart extraction and linking
   - Interactive table exploration

---

## References

### Research Sources

1. **PDF Table Extraction**:
   - [Python Libraries to Extract Tables From PDF: A Comparison](https://unstract.com/blog/extract-tables-from-pdf-python/)
   - [I Tested 7 Python PDF Extractors So You Don't Have To (2025 Edition)](https://onlyoneaman.medium.com/i-tested-7-python-pdf-extractors-so-you-dont-have-to-2025-edition-c88013922257)
   - [Camelot: PDF Table Extraction for Humans](https://github.com/atlanhq/camelot)
   - [pdfplumber Documentation](https://github.com/jsvine/pdfplumber)
   - [How to Extract PDF Tables in Python? - GeeksforGeeks](https://www.geeksforgeeks.org/python/how-to-extract-pdf-tables-in-python/)

2. **RAG with Structured Data**:
   - [RAG for Structured Data: Benefits, Challenges & Examples](https://www.ai21.com/knowledge/rag-for-structured-data/)
   - [Best practices for structuring large datasets in RAG](https://www.datasciencecentral.com/best-practices-for-structuring-large-datasets-in-retrieval-augmented-generation-rag/)
   - [Using a Knowledge Graph to Implement a RAG Application](https://www.datacamp.com/tutorial/knowledge-graph-rag)
   - [RAG Tutorial: How to Build a RAG System on a Knowledge Graph](https://neo4j.com/blog/developer/rag-tutorial/)
   - [GraphRAG Explained: Enhancing RAG with Knowledge Graphs](https://medium.com/@zilliz_learn/graphrag-explained-enhancing-rag-with-knowledge-graphs-3312065f99e1)

3. **Knowledge Graph Integration**:
   - [Chapter 1 — How to Build Accurate RAG Over Structured and Semi-structured Databases](https://medium.com/madhukarkumar/chapter-1-how-to-build-accurate-rag-over-structured-and-semi-structured-databases-996c68098dba)
   - [RAG using structured data: Overview & important questions](https://blog.kuzudb.com/post/llms-graphs-part-1/)
   - [Implementing Graph RAG Using Knowledge Graphs](https://www.ibm.com/think/tutorials/knowledge-graph-rag)

### Library Documentation

- [pdfplumber](https://github.com/jsvine/pdfplumber): PDF text and table extraction
- [Camelot](https://camelot-py.readthedocs.io/): Specialized PDF table extraction
- [ChromaDB](https://docs.trychroma.com/): Vector database
- [NetworkX](https://networkx.org/): Knowledge graph management
- [Pandas](https://pandas.pydata.org/): Data manipulation and analysis

---

## Appendix

### ChromaDB Metadata Limitations

- Metadata values must be JSON-serializable
- Nested objects are stored as JSON strings
- Large nested structures may impact performance
- Consider external storage for large tables (Phase 3)

### pdfplumber Table Extraction Settings

Default settings work well for most tables. Advanced options:

```python
tables = page.extract_tables({
    "vertical_strategy": "lines",     # or "text", "explicit"
    "horizontal_strategy": "lines",   # or "text", "explicit"
    "snap_tolerance": 3,              # Pixels for line snapping
    "join_tolerance": 3,              # Pixels for joining lines
    "edge_min_length": 3,             # Minimum edge length
    "min_words_vertical": 3,          # Min words for vertical detection
    "min_words_horizontal": 1,        # Min words for horizontal detection
})
```

### Migration Path

When implementing future phases:

1. **Phase 2 Migration**:
   - No data migration needed
   - Add config options without breaking existing setup
   - Reindex documents to populate new metadata fields

2. **Phase 3 Migration**:
   - Export existing tables from vector store
   - Import to new table store
   - Update references in metadata
   - Test dual storage retrieval

3. **Phase 4 Migration**:
   - Extract entities from existing tables
   - Build table nodes in knowledge graph
   - Link to existing document/entity nodes
   - Validate graph consistency

---

**Document Status**: Living document, updated as implementation progresses.

**Maintainers**: Development team

**Review Cycle**: After each phase completion
