import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any
import logging
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, track
from rich.panel import Panel
from rich.prompt import Prompt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from ..utils.config import Config
from ..processors.pdf_processor import PDFProcessor
from ..processors.image_processor import ImageProcessor
from ..processors.pptx_processor import PPTXProcessor
from ..processors.docx_processor import DOCXProcessor
from ..processors.text_processor import TextProcessor
from ..database.vector_store import VectorStore
from ..rag.retriever import DocumentRetriever
from ..rag.generator import ResponseGenerator
from ..knowledge_graph.kg_manager import KnowledgeGraphManager
from ..knowledge_graph.entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)
console = Console()

class RAGCLI:
    """Command line interface for the RAG system"""
    
    def __init__(self, config_path="config.yaml"):
        """Initialize CLI with configuration"""
        self.config = Config(config_path)
        self.config.ensure_paths()
        
        # Initialize components
        self.vector_store = None
        self.retriever = None
        self.generator = None
        self.knowledge_graph = None
        self.entity_extractor = None
        self.processors = {
            'pdf': PDFProcessor(),
            'image': ImageProcessor(
                language=self.config.get('ocr.language', 'eng'),
                config=self.config.get('ocr.config', '--psm 6')
            ),
            'pptx': PPTXProcessor(),
            'docx': DOCXProcessor(),
            'txt': TextProcessor()
        }
    
    def _initialize_rag_components(self):
        """Initialize RAG components if not already done"""
        if not self.vector_store:
            self.vector_store = VectorStore(
                db_path=self.config.get('database.path'),
                collection_name=self.config.get('database.collection_name'),
                embedding_model=self.config.get('database.embedding_model')
            )
            
            # Initialize knowledge graph components
            self.knowledge_graph = KnowledgeGraphManager()
            self.entity_extractor = EntityExtractor()
            
            self.retriever = DocumentRetriever(
                self.vector_store,
                self.knowledge_graph,
                self.entity_extractor
            )
            
            self.generator = ResponseGenerator(
                model_name=self.config.get('llm.model'),
                temperature=self.config.get('llm.temperature'),
                max_tokens=self.config.get('llm.max_tokens')
            )
    
    def index_command(self, args):
        """Index documents in the specified directory"""
        self._initialize_rag_components()
        
        data_path = Path(args.path or self.config.get('documents.data_path'))
        
        if not data_path.exists():
            console.print(f"[red]Error: Path {data_path} does not exist[/red]")
            return
        
        # Find all supported files
        files_to_process = self._find_files(data_path)
        
        if not files_to_process:
            console.print("[yellow]No supported files found[/yellow]")
            return
        
        console.print(f"[green]Found {len(files_to_process)} files to process[/green]")
        
        # Process files
        all_documents = []
        
        with Progress() as progress:
            task = progress.add_task("Processing files...", total=len(files_to_process))
            
            for file_path in files_to_process:
                try:
                    # Determine file type
                    file_type = self._get_file_type(file_path)
                    processor = self.processors.get(file_type)
                    
                    if processor and processor.is_supported(file_path):
                        # Process the file
                        documents = processor.process(file_path)
                        all_documents.extend(documents)
                        
                    progress.advance(task)
                    
                except Exception as e:
                    console.print(f"[red]Error processing {file_path}: {str(e)}[/red]")
                    progress.advance(task)
        
        if all_documents:
            # Show indexing progress
            console.print(f"[green]🔍 Processing {len(files_to_process)} files...[/green]")
            
            # Add documents to vector store
            console.print("[blue]📚 Adding documents to vector store...[/blue]")
            success = self.vector_store.add_documents(all_documents)
            
            if success:
                # Extract entities and build knowledge graph
                console.print("[blue]🧠 Building knowledge graph from documents...[/blue]")
                self._build_knowledge_graph(all_documents)
                
                # Save knowledge graph
                self.knowledge_graph.save_graph()
                
                # Show results summary
                stats = self.vector_store.get_collection_stats()
                kg_stats = self.knowledge_graph.get_graph_stats()
                
                console.print(f"\n[green]✅ Successfully indexed {len(files_to_process)} files![/green]")
                console.print(f"[blue]📚 Vector Store: {stats.get('document_count', 0)} documents[/blue]")
                console.print(f"[blue]🧠 Knowledge Graph: {kg_stats.get('entity_count', 0)} entities, {kg_stats.get('total_edges', 0)} relationships[/blue]")
                
                # Show entity type distribution
                entity_types = kg_stats.get('entity_types', {})
                if entity_types:
                    console.print("[blue]📊 Entity Types:[/blue]")
                    for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
                        console.print(f"  • {entity_type}: {count}")
                
                # Add some fun stats
                total_entities = kg_stats.get('entity_count', 0)
                if total_entities > 0:
                    avg_entities_per_doc = total_entities / len(files_to_process)
                    console.print(f"[blue]📈 Average: {avg_entities_per_doc:.1f} entities per document[/blue]")
                
                if len(files_to_process) > 1:
                    console.print(f"[blue]🎯 Ready for intelligent search across {len(files_to_process)} documents![/blue]")
                    
            else:
                console.print("[red]❌ Failed to add documents to vector store[/red]")
        else:
            console.print("[yellow]No documents extracted from files[/yellow]")
    
    def search_command(self, args):
        """Search for documents matching a query"""
        self._initialize_rag_components()
        
        if not args.query:
            args.query = Prompt.ask("Enter your search query")
        
        # Retrieve documents
        documents = self.retriever.retrieve(
            query=args.query,
            top_k=args.top_k or self.config.get('retrieval.top_k'),
            similarity_threshold=args.threshold or self.config.get('retrieval.similarity_threshold')
        )
        
        if not documents:
            console.print("[yellow]No documents found matching your query[/yellow]")
            return
        
        # Display results
        self._display_search_results(documents, args.query)
    
    def query_command(self, args):
        """Query the RAG system for an answer"""
        self._initialize_rag_components()
        
        if not self.generator.is_available():
            console.print("[red]Error: Ollama is not available or the model is not installed[/red]")
            console.print("[yellow]Please ensure Ollama is running and the model is installed[/yellow]")
            console.print(f"[yellow]Run: ollama pull {self.config.get('llm.model')}[/yellow]")
            return
        
        if not args.query:
            args.query = Prompt.ask("Enter your question")
        
        with console.status("Thinking...", spinner="dots"):
            # Retrieve relevant documents
            context = self.retriever.get_context_for_query(args.query)
            
            # Generate response
            response = self.generator.generate_response(
                query=args.query,
                context=context
            )
        
        # Display response
        console.print("\n[bold]Question:[/bold]")
        console.print(f"[blue]{args.query}[/blue]\n")
        
        console.print("[bold]Answer:[/bold]")
        console.print(Panel(response.expandtabs(2), title="Generated Response"))
        
        if args.show_context:
            console.print("\n[bold]Context:[/bold]")
            console.print(Panel(context.expandtabs(2), title="Retrieved Context"))
    
    def stats_command(self, args):
        """Show collection statistics"""
        self._initialize_rag_components()
        
        stats = self.vector_store.get_collection_stats()
        
        if stats:
            # Create a table for stats
            table = Table(title="Vector Store Statistics")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Collection Name", stats.get('name', 'N/A'))
            table.add_row("Document Count", str(stats.get('document_count', 0)))
            table.add_row("Embedding Dimension", str(stats.get('embedding_dimension', 'N/A')))
            table.add_row("Storage Path", stats.get('path', 'N/A'))
            
            console.print(table)
        else:
            console.print("[red]Error retrieving collection statistics[/red]")
    
    def reset_command(self, args):
        """Reset the vector store"""
        if not args.force:
            confirm = Prompt.ask(
                "This will delete all documents from the vector store. Are you sure?",
                choices=["y", "n"],
                default="n"
            )
            
            if confirm.lower() != "y":
                console.print("[yellow]Operation cancelled[/yellow]")
                return
        
        self._initialize_rag_components()
        self.vector_store.reset_store()
        console.print("[green]Vector store reset successfully[/green]")
    
    def clear_command(self, args):
        """Clear all data and start fresh"""
        if not args.force:
            message = "⚠️  This will delete ALL data and files (vector store + knowledge graph + knowledge_graph.json). Are you absolutely sure?"
            confirm = Prompt.ask(
                message,
                choices=["y", "n"],
                default="n"
            )
            
            if confirm.lower() != "y":
                console.print("[yellow]Operation cancelled[/yellow]")
                return
        
        console.print("[red]🗑️  Clearing all system data...[/red]")
        
        try:
            # Delete vector database directory
            import shutil
            vector_db_path = Path(self.config.get('database.path'))
            if vector_db_path.exists():
                shutil.rmtree(vector_db_path)
                console.print("[green]✅ Vector database deleted[/green]")
            
            # Delete knowledge graph file
            kg_file = Path("knowledge_graph.json")
            if kg_file.exists():
                kg_file.unlink()
                console.print("[green]✅ Knowledge graph file deleted[/green]")
            
            console.print("[green]🎯 Complete system reset successful![/green]")
            console.print("[blue]💡 Ready for fresh indexing![/blue]")
            
        except Exception as e:
            console.print(f"[red]❌ Error during clear operation: {str(e)}[/red]")
    
    def _find_files(self, directory: Path) -> List[Path]:
        """Find all supported files in directory"""
        supported_formats = self.config.get('documents.supported_formats', [])
        files = []
        
        for format_ext in supported_formats:
            pattern = f"*.{format_ext}"
            files.extend(directory.rglob(pattern))
        
        return sorted(files)
    
    def _get_file_type(self, file_path: Path) -> str:
        """Determine file type based on extension"""
        ext = file_path.suffix.lower()
        
        if ext == '.pdf':
            return 'pdf'
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp']:
            return 'image'
        elif ext in ['.pptx', '.ppt']:
            return 'pptx'
        elif ext in ['.docx', '.doc']:
            return 'docx'
        elif ext in ['.txt']:
            return 'txt'
        else:
            return 'unknown'
    
    def _display_search_results(self, documents: List[Dict[str, Any]], query: str):
        """Display search results in a formatted table"""
        table = Table(title=f"Search Results for: {query}")
        table.add_column("Score", style="cyan", width=8)
        table.add_column("Source", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Content Preview", style="white")
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            source = Path(metadata.get('source', 'Unknown')).name
            file_type = metadata.get('file_type', 'Unknown')
            similarity = f"{doc.get('similarity', 0):.3f}"
            
            # Truncate content for display
            content = doc.get('text', '')[:150]
            if len(doc.get('text', '')) > 150:
                content += "..."
            
            table.add_row(similarity, source, file_type, content)
        
        console.print(table)
    
    def interactive_mode(self):
        """Start interactive mode"""
        self._initialize_rag_components()
        
        console.print("[bold green]Welcome to RAG Interactive Mode![/bold green]")
        console.print("Type 'help' for available commands or 'quit' to exit.\n")
        
        while True:
            try:
                command = Prompt.ask("[bold blue]RAG>[/bold blue]").strip()
                
                if not command:
                    continue
                
                if command.lower() in ['quit', 'exit', 'q']:
                    console.print("[green]Goodbye![/green]")
                    break
                
                if command.lower() == 'help':
                    self._show_interactive_help()
                    continue
                
                # Treat as query
                with console.status("Thinking...", spinner="dots"):
                    context = self.retriever.get_context_for_query(command)
                    response = self.generator.generate_response(query=command, context=context)
                
                console.print(Panel(response.expandtabs(2), title="Response"))
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'quit' to exit[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
    
    def _show_interactive_help(self):
        """Show help for interactive mode"""
        help_text = """
Available commands:
- Any text: Ask a question (will search and generate response)
- help: Show this help message
- quit/exit/q: Exit interactive mode

Examples:
- What are the key points in the PDF documents?
- Summarize the information about machine learning
- Find documents related to data analysis
        """
        console.print(Panel(help_text.strip(), title="Interactive Mode Help"))
    
    def _build_knowledge_graph(self, all_documents: List[Dict[str, Any]]):
        """Build knowledge graph from processed documents"""
        try:
            # Load existing knowledge graph
            self.knowledge_graph.load_graph()
            
            # Group documents by source file
            doc_groups = defaultdict(list)
            for doc in all_documents:
                source = doc.get('metadata', {}).get('source', 'unknown')
                doc_groups[source].append(doc)
            
            # Process each document for entities
            for doc_id, doc_chunks in doc_groups.items():
                # Combine all text from chunks
                full_text = ' '.join(chunk['text'] for chunk in doc_chunks)
                
                # Extract entities
                entities = self.entity_extractor.extract_entities(full_text, doc_id)
                
                # Add to knowledge graph
                self.knowledge_graph.add_document_entities(doc_id, full_text, entities)
                
                console.print(f"  [cyan]📝 Processed {Path(doc_id).name}: {len(entities)} entities discovered[/cyan]")
            
        except Exception as e:
            console.print(f"[red]Error building knowledge graph: {str(e)}[/red]")
    
    def kg_search_command(self, args):
        """Search the knowledge graph for entities"""
        self._initialize_rag_components()
        
        # Explicitly load the knowledge graph
        self.knowledge_graph.load_graph()
        
        if not args.query:
            args.query = Prompt.ask("Enter entity to search for")
        
        # Search for entities
        entities = self.knowledge_graph.search_entities(args.query, limit=args.limit or 10)
        
        if not entities:
            console.print("[yellow]No entities found matching your query[/yellow]")
            return
        
        # Display results
        table = Table(title=f"Knowledge Graph Entities for: {args.query}")
        table.add_column("Entity", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Document Count", style="yellow")
        table.add_column("Relevance", style="red")
        
        for entity in entities:
            table.add_row(
                entity['entity'],
                entity['type'],
                str(entity['document_count']),
                f"{entity['relevance']:.2f}"
            )
        
        console.print(table)
    
    def kg_related_command(self, args):
        """Find entities related to a given entity"""
        self._initialize_rag_components()
        
        # Explicitly load the knowledge graph
        self.knowledge_graph.load_graph()
        
        if not args.entity:
            args.entity = Prompt.ask("Enter entity name")
        
        # Find related entities
        related = self.knowledge_graph.find_related_entities(
            args.entity, 
            max_depth=args.depth or 2
        )
        
        if not related:
            console.print(f"[yellow]No entities found related to '{args.entity}'[/yellow]")
            return
        
        # Display results
        table = Table(title=f"🔗 Entities Related to: {args.entity}")
        table.add_column("Entity", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Distance", style="yellow")
        table.add_column("Relationship", style="blue")
        table.add_column("Documents", style="magenta")
        
        # Show top related entities for readability
        for rel in related[:20]:  # Limit to top 20
            rel_type = rel['relationship_info'].get('relationship', 'related')
            doc_count = len(rel['relationship_info'].get('documents', []))
            entity_name = rel['entity'][:50] + ('...' if len(rel['entity']) > 50 else '')
            table.add_row(
                entity_name,
                rel['entity_type'],
                str(rel['distance']),
                rel_type,
                str(doc_count)
            )
        
        console.print(table)
        
        if len(related) > 20:
            console.print(f"[blue]... and {len(related) - 20} more related entities (use --depth to explore deeper)[/blue]")
    
    def kg_stats_command(self, args):
        """Show knowledge graph statistics"""
        self._initialize_rag_components()
        
        # Explicitly load the knowledge graph
        self.knowledge_graph.load_graph()
        
        stats = self.knowledge_graph.get_graph_stats()
        
        if stats:
            # Create a table for stats
            table = Table(title="Knowledge Graph Statistics")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Nodes", str(stats.get('total_nodes', 0)))
            table.add_row("Total Edges", str(stats.get('total_edges', 0)))
            table.add_row("Entity Count", str(stats.get('entity_count', 0)))
            table.add_row("Document Count", str(stats.get('document_count', 0)))
            table.add_row("Avg Connections", f"{stats.get('avg_connections', 0):.2f}")
            
            console.print(table)
            
            # Show entity types
            entity_types = stats.get('entity_types', {})
            if entity_types:
                console.print("\n[blue]Entity Types:[/blue]")
                for entity_type, count in entity_types.items():
                    console.print(f"  {entity_type}: {count}")
        else:
            console.print("[red]Error retrieving knowledge graph statistics[/red]")

def create_parser():
    """Create the argument parser"""
    parser = argparse.ArgumentParser(
        description="RAG (Retrieval-Augmented Generation) CLI for local document processing"
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to configuration file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument(
        "--path", "-p",
        help="Path to directory containing documents"
    )
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for documents")
    search_parser.add_argument("query", nargs="?", help="Search query")
    search_parser.add_argument("--top-k", type=int, help="Number of results to return")
    search_parser.add_argument("--threshold", type=float, help="Similarity threshold")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("query", nargs="?", help="Question to ask")
    query_parser.add_argument("--show-context", action="store_true", help="Show retrieved context")
    
    # Stats command
    subparsers.add_parser("stats", help="Show collection statistics")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset the vector store")
    reset_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    reset_parser.add_argument("--clear", action="store_true", help="Also clear knowledge graph")
    
    # Clear command (complete reset)
    clear_parser = subparsers.add_parser("clear", help="Clear all data and start fresh")
    clear_parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    
    # Interactive mode
    subparsers.add_parser("interactive", help="Start interactive mode")
    
    # Knowledge Graph commands
    kg_subparsers = subparsers.add_parser("kg", help="Knowledge graph commands")
    kg_parsers = kg_subparsers.add_subparsers(dest="kg_command", help="KG commands")
    
    # KG search
    kg_search_parser = kg_parsers.add_parser("search", help="Search knowledge graph")
    kg_search_parser.add_argument("query", nargs="?", help="Entity to search for")
    kg_search_parser.add_argument("--limit", type=int, help="Number of results")
    
    # KG related entities
    kg_related_parser = kg_parsers.add_parser("related", help="Find related entities")
    kg_related_parser.add_argument("entity", nargs="?", help="Entity name")
    kg_related_parser.add_argument("--depth", type=int, help="Search depth")
    
    # KG stats
    kg_parsers.add_parser("stats", help="Show knowledge graph statistics")
    
    return parser

def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    try:
        cli = RAGCLI(args.config)
        
        # Execute command
        if args.command == "index":
            cli.index_command(args)
        elif args.command == "search":
            cli.search_command(args)
        elif args.command == "query":
            cli.query_command(args)
        elif args.command == "stats":
            cli.stats_command(args)
        elif args.command == "reset":
            cli.reset_command(args)
        elif args.command == "clear":
            cli.clear_command(args)
        elif args.command == "clear":
            cli.clear_command(args)
        elif args.command == "interactive":
            cli.interactive_mode()
        elif args.command == "kg":
            if args.kg_command == "search":
                cli.kg_search_command(args)
            elif args.kg_command == "related":
                cli.kg_related_command(args)
            elif args.kg_command == "stats":
                cli.kg_stats_command(args)
            else:
                console.print("[red]Please specify a KG command: search, related, or stats[/red]")
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        if args.verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)
