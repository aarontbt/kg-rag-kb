# 🧠 Knowledge Graph-Enhanced RAG System

A powerful local Retrieval-Augmented Generation (RAG) system with integrated knowledge graph capabilities. This system processes multiple document formats, automatically extracts entities and relationships, and provides intelligent search and question-answering capabilities.

## ✨ Key Features

- 🔄 **Multi-format document processing** (PDF, Images, PowerPoint, Word, Text)
- 🧠 **Knowledge graph integration** (automatic entity and relationship extraction)
- 🔍 **Hybrid search** (vector similarity + graph relationships)
- 💬 **LLM-powered answers** (with Ollama integration)
- 📊 **Entity discovery** (people, organizations, technologies, concepts, locations)
- 🔗 **Relationship mapping** (find connections between concepts)
- 💻 **Rich CLI interface** with beautiful terminal output
- 🚀 **Local operation** (no external API dependencies)

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Tesseract OCR** (for image processing):
   ```bash
   # macOS
   brew install tesseract

   # Ubuntu
   sudo apt-get install tesseract-ocr
   ```

3. **Ollama** (for LLM responses, optional):
   ```bash
   # Install Ollama
   curl https://ollama.ai/install.sh | sh

   # Pull a model
   ollama pull llama2
   ```

### Installation

1. **Clone/Download the project**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

1. **Add your documents** to the `data/` folder:
   - PDF files
   - Images (.png, .jpg, .jpeg)
   - PowerPoint files (.pptx, .ppt)
   - Word documents (.docx, .doc)
   - Text files (.txt)

2. **Index documents** (builds vector store + knowledge graph):
   ```bash
   python main.py index
   ```

3. **Start using the system!**

## 📚 Command Reference

### 🔍 Document Search

```bash
# Basic search
python main.py search "machine learning"

# Search with custom parameters
python main.py search "artificial intelligence" --top-k 10 --threshold 0.1
```

### 💬 Ask Questions

```bash
# Ask questions using your documents
python main.py query "What is machine learning?"

# Show retrieved context
python main.py query "What are the applications of AI?" --show-context
```

### 🧠 Knowledge Graph Commands

```bash
# View graph statistics
python main.py kg stats

# Search for entities
python main.py kg search "AI"
python main.py kg search "machine learning" --limit 10

# Find related entities (powerful!)
python main.py kg related "AI" --depth 2
python main.py kg related "computer vision"
```

### 📊 System Statistics

```bash
# Vector store statistics
python main.py stats

# Knowledge graph statistics
python main.py kg stats
```

### 🔄 System Management

```bash
# Interactive mode (chat with your documents)
python main.py interactive

# Reset the system
python main.py reset --force

# View all commands
python main.py --help
python main.py kg --help
```

## 🎯 Use Cases

### 🔬 Research & Academia

```bash
# 1. Add research papers
python main.py index

# 2. Find related concepts
python main.py kg related "machine learning" --depth 3

# 3. Ask research questions
python main.py query "How does deep learning relate to neural networks?"
```

### 💼 Business & Knowledge Management

```bash
# 1. Add business documents
python main.py index

# 2. Explore entity relationships
python main.py kg related "product name"

# 3. Ask business questions
python main.py query "What are our competitive advantages?"
```

### 📚 Learning & Education

```bash
# 1. Add study materials
python main.py index

# 2. Discover connections
python main.py kg related "key concept" --depth 2

# 3. Get explanations
python main.py query "Explain this concept in simple terms"
```

## 🧠 Knowledge Graph Capabilities

### Entity Types Automatically Detected

| Type | Examples | Description |
|------|----------|-------------|
| **Person** | "John Doe", "Dr. Smith" | Names of people |
| **Organization** | "Google", "MIT" | Companies, institutions |
| **Technology** | "Python", "TensorFlow" | Software, frameworks |
| **Concept** | "Machine Learning", "AI" | Abstract ideas |
| **Location** | "California", "New York" | Geographical places |

### Relationship Types Identified

- **`is_type`** - Classification relationships
- **`part_of`** - Component relationships
- **`created`** - Creator relationships
- **`uses`** - Tool/technology relationships
- **`located_in`** - Location relationships

### Example Knowledge Graph Output

```
Knowledge Graph Statistics
┏━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Property        ┃ Value ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ Total Nodes     │ 161   │
│ Total Edges     │ 160   │
│ Entity Count    │ 160   │
│ Document Count  │ 1     │
│ Avg Connections │ 1.00  │
└─────────────────┴───────┘

Entity Types:
  person: 122
  technology: 2
  location: 35
  concept: 1
```

## ⚙️ Configuration

Edit `config.yaml` to customize the system:

```yaml
# Database settings
database:
  path: "./vector_db"
  collection_name: "documents"
  embedding_model: "all-MiniLM-L6-v2"

# Document processing
documents:
  data_path: "./data"
  supported_formats: ["pdf", "png", "jpg", "jpeg", "pptx", "docx", "txt"]
  chunk_size: 1000
  chunk_overlap: 200

# OCR settings
ocr:
  language: "eng"
  config: "--psm 6"

# Retrieval settings
retrieval:
  top_k: 5
  similarity_threshold: 0.1

# LLM settings (Ollama)
llm:
  model: "llama2"  # Change to qwen3:8b, llama3.2:1b, etc.
  temperature: 0.7
  max_tokens: 1000
```

## 📁 Project Structure

```
rainmarket-chatbot/
├── src/
│   ├── processors/          # Document processors
│   │   ├── pdf_processor.py
│   │   ├── image_processor.py
│   │   ├── pptx_processor.py
│   │   ├── docx_processor.py
│   │   └── text_processor.py
│   ├── database/           # Vector database
│   │   ├── vector_store.py
│   │   └── embeddings.py
│   ├── knowledge_graph/   # Knowledge graph
│   │   ├── kg_manager.py
│   │   └── entity_extractor.py
│   ├── rag/               # RAG pipeline
│   │   ├── retriever.py
│   │   └── generator.py
│   ├── cli/               # Command line interface
│   │   └── commands.py
│   └── utils/             # Utilities
│       └── config.py
├── data/                  # Document storage
├── vector_db/            # ChromaDB storage
├── knowledge_graph.json  # Knowledge graph data
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies
└── main.py             # Entry point
```

## 🔧 Advanced Features

### Entity Extraction Enhancement

The system uses rule-based extraction with optional spaCy integration:

```python
# Install spaCy for better NER (optional)
pip install spacy
python -m spacy download en_core_web_sm
```

### Custom Entity Patterns

Create `custom_patterns.json` to add domain-specific patterns:

```json
{
  "technology": [
    "your_tech_term",
    "another_technology"
  ],
  "organization": [
    "your_company",
    "specific_institution"
  ]
}
```

### Knowledge Graph Visualization

Export subgraph data for external visualization:

```python
# The system can export visualization data for tools like:
# - Gephi
# - Cytoscape
# - D3.js visualizations
```

## 🐛 Troubleshooting

### Common Issues

1. **"spaCy model not found"**
   - This is normal! The system uses rule-based extraction
   - Install spaCy for better accuracy: `pip install spacy && python -m spacy download en_core_web_sm`

2. **"Ollama API error"**
   - Ensure Ollama is running: `ollama list`
   - Check model installation: `ollama pull llama2`

3. **"Tesseract not found"**
   - Install Tesseract OCR for image processing
   - macOS: `brew install tesseract`
   - Ubuntu: `sudo apt-get install tesseract-ocr`

4. **"No entities found"**
   - Check if documents were processed: `python main.py index`
   - Verify document formats are supported

### Performance Tips

- **Large documents**: Reduce `chunk_size` in config.yaml
- **Many entities**: Increase `similarity_threshold` for better filtering
- **Memory usage**: Use smaller LLM models like `llama3.2:1b`

## 🆚 Comparison with Alternatives

| Feature | This System | Memgraph | Neo4j | ChromaDB Only |
|---------|-------------|----------|--------|---------------|
| **Setup Complexity** | ⭐ Easy | ⭐⭐⭐ Docker | ⭐⭐⭐ Complex | ⭐ Easy |
| **Storage Size** | ⭐ Small | ⭐⭐⭐ Large | ⭐⭐⭐ Large | ⭐ Small |
| **Entity Extraction** | ✅ Automatic | ❌ Manual | ❌ Manual | ❌ None |
| **Relationship Discovery** | ✅ Automatic | ❌ Manual | ❌ Manual | ❌ None |
| **LLM Integration** | ✅ Built-in | ❌ External | ❌ External | ❌ None |
| **Vector Search** | ✅ Built-in | ❌ External | ❌ External | ✅ Built-in |
| **Local Operation** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Customizable** | ✅ Easy | ⭐⭐ Medium | ⭐⭐ Medium | ✅ Easy |

## 🎯 When to Use This System

### ✅ **Perfect For**
- **Research papers analysis** - Find connections between concepts
- **Knowledge management** - Organize and explore your documents
- **Learning systems** - Discover relationships between topics
- **Document-based chatbots** - Build intelligent assistants
- **Content discovery** - Find hidden connections in large document sets

### ❌ **Not Ideal For**
- **Real-time graph processing** - Use dedicated graph databases
- **Large-scale enterprise deployment** - Use Neo4j/Memgraph
- **High-frequency updates** - NetworkX has performance limits
- **Complex graph algorithms** - Use specialized graph databases

## 🤝 Contributing

This system is designed to be easily extensible:

1. **Add new processors** for additional file formats
2. **Enhance entity extraction** with custom patterns
3. **Improve relationship detection** with NLP models
4. **Add visualization tools** for graph exploration
5. **Integrate additional LLMs** beyond Ollama

## 📄 License

This project is open source and available under the MIT License.