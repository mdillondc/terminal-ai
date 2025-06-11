import os
import time
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import ollama

from document_processor import DocumentProcessor
from embedding_service import EmbeddingService
from vector_store import VectorStore
from settings_manager import SettingsManager


class RAGEngine:
    def __init__(self, client: OpenAI):
        self.client = client
        self.settings_manager = SettingsManager.getInstance()
        
        # Initialize Ollama client
        ollama_url = self.settings_manager.setting_get("ollama_base_url")
        try:
            self.ollama_client = ollama.Client(host=ollama_url)
        except Exception:
            self.ollama_client = None
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService(client, self.ollama_client)
        self.vector_store = VectorStore()
        
        # Active collection state
        self.active_collection = None
        self.active_collection_chunks = None
        
        # Load active collection from settings if available
        saved_collection = self.settings_manager.setting_get("rag_active_collection")
        if saved_collection and self.collection_exists(saved_collection):
            self.activate_collection(saved_collection, verbose=False)
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection directory exists and has supported files"""
        available_collections = self.vector_store.get_available_collections()
        return collection_name in available_collections
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """Get list of available collections with their info"""
        collections = []
        available_names = self.vector_store.get_available_collections()
        
        for name in available_names:
            info = self.vector_store.get_collection_info(name)
            collections.append(info)
        
        return collections
    
    def build_collection(self, collection_name: str, force_rebuild: bool = False) -> bool:
        """
        Build/rebuild embeddings for a collection
        
        Args:
            collection_name: Name of the collection to build
            force_rebuild: If True, rebuild even if cache is valid
            
        Returns:
            True if successful, False otherwise
        """
        if not self.collection_exists(collection_name):
            print(f"- Collection '{collection_name}' not found")
            return False
        
        # Check if rebuild is needed
        if not force_rebuild and self.vector_store.is_collection_cache_valid(collection_name):
            print(f"- Collection '{collection_name}' index is up to date")
            return True
        
        print(f"- Building embeddings for collection '{collection_name}'...")
        
        try:
            # Get collection path
            collections_path = os.path.join(self.settings_manager.setting_get("working_dir"), "rag")
            collection_path = os.path.join(collections_path, collection_name)
            
            # Process documents into chunks
            print("- Processing documents...")
            chunks = self.document_processor.process_collection(collection_path)
            
            if not chunks:
                print("- No processable documents found in collection")
                return False
            
            # Generate embeddings
            print(f"- Generating embeddings for {len(chunks)} chunks...")
            chunk_texts = [chunk["content"] for chunk in chunks]
            
            embeddings = self.embedding_service.generate_embeddings_batch(chunk_texts)
            
            if len(embeddings) != len(chunks):
                print("- Mismatch between chunks and embeddings")
                return False
            
            # Add embeddings to chunks
            for i, embedding in enumerate(embeddings):
                chunks[i]["embedding"] = embedding
                chunks[i]["created_at"] = time.time()
            
            # Save to vector store
            print("- Saving index...")
            success = self.vector_store.save_collection_index(collection_name, chunks)
            
            if success:
                print(f"- Successfully built collection '{collection_name}'")
                
                # If this is the active collection, reload it
                if self.active_collection == collection_name:
                    self.active_collection_chunks = chunks
                
                return True
            else:
                print(f"- Failed to save collection '{collection_name}'")
                return False
                
        except Exception as e:
            print(f"- Error building collection '{collection_name}': {e}")
            return False
    
    def activate_collection(self, collection_name: str, verbose: bool = True) -> bool:
        """
        Activate a collection for RAG queries
        
        Args:
            collection_name: Name of collection to activate
            verbose: Whether to print status messages
            
        Returns:
            True if successful, False otherwise
        """
        if not self.collection_exists(collection_name):
            if verbose:
                print(f"- Collection '{collection_name}' not found")
            return False
        
        # Check if index exists
        index_exists = os.path.exists(self.vector_store._get_index_file_path(collection_name))
        cache_valid = self.vector_store.is_collection_cache_valid(collection_name)
        
        if not index_exists:
            if verbose:
                print(f"- Collection '{collection_name}' has no index")
                print(f"- Run: --rag-build {collection_name}")
            return False
        
        if not cache_valid:
            if verbose:
                print(f"- Collection '{collection_name}' files have changed since last build")
                print(f"- Auto-rebuilding collection...")
            
            # Auto-rebuild the collection
            success = self.build_collection(collection_name, force_rebuild=True)
            if not success:
                if verbose:
                    print(f"- Failed to rebuild collection '{collection_name}'")
                return False
        
        # Load collection index
        chunks = self.vector_store.load_collection_index(collection_name)
        if chunks is None:
            if verbose:
                print(f"- Failed to load collection '{collection_name}'")
            return False
        
        # Activate collection
        self.active_collection = collection_name
        self.active_collection_chunks = chunks
        
        # Save to settings
        self.settings_manager.setting_set("rag_active_collection", collection_name)
        
        if verbose:
            print(f"- Activated collection '{collection_name}' ({len(chunks)} chunks)")
        
        return True
    
    def deactivate_collection(self) -> None:
        """Deactivate the current collection"""
        if self.active_collection:
            print(f"- Deactivated collection '{self.active_collection}'")
        
        self.active_collection = None
        self.active_collection_chunks = None
        self.settings_manager.setting_set("rag_active_collection", None)
    
    def query(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query the active collection for relevant chunks
        
        Args:
            query_text: The search query
            top_k: Number of results to return (defaults to setting)
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if not self.active_collection or not self.active_collection_chunks:
            return []
        
        if not query_text.strip():
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query_text)
            
            # Find similar chunks
            results = self.embedding_service.find_most_similar(
                query_embedding, 
                self.active_collection_chunks, 
                top_k
            )
            
            return results
            
        except Exception as e:
            print(f"- Error querying collection: {e}")
            return []
    
    def get_context_for_query(self, query_text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get formatted context string for AI and source information
        
        Args:
            query_text: The user's query
            
        Returns:
            Tuple of (formatted_context, source_chunks)
        """
        if not self.is_active():
            return "", []
        
        # Get relevant chunks
        relevant_chunks = self.query(query_text)
        
        if not relevant_chunks:
            return "", []
        
        # Calculate context management settings
        context_strategy = self.settings_manager.setting_get("context_management_strategy")
        max_context_tokens = self._get_context_budget(context_strategy)
        
        # Format context
        context_parts = []
        current_tokens = 0
        used_chunks = []
        
        context_header = f"You have access to relevant information from the user's document collection '{self.active_collection}':\n\nCONTEXT:\n---\n"
        current_tokens += self.embedding_service.count_tokens(context_header)
        
        for chunk in relevant_chunks:
            chunk_text = f"[From {chunk['filename']}]\n{chunk['content']}\n\n"
            chunk_tokens = self.embedding_service.count_tokens(chunk_text)
            
            if current_tokens + chunk_tokens > max_context_tokens:
                break
            
            context_parts.append(chunk_text)
            current_tokens += chunk_tokens
            used_chunks.append(chunk)
        
        if not context_parts:
            return "", []
        
        context_footer = "---\n\nUse this context to enhance your responses. Cite sources when referencing specific information."
        
        full_context = context_header + "".join(context_parts) + context_footer
        
        return full_context, used_chunks
    
    def _get_context_budget(self, strategy: str) -> int:
        """Get context token budget based on strategy"""
        budgets = {
            "generous": 10000,
            "balanced": 5000,
            "strict": 2000
        }
        return budgets.get(strategy, 5000)
    
    def _check_collection_needs_rebuild(self, collection_name: str) -> bool:
        """Check if a collection needs rebuilding due to file changes"""
        try:
            # Get file metadata
            collections_path = os.path.join(self.settings_manager.setting_get("working_dir"), "rag")
            collection_path = os.path.join(collections_path, collection_name)
            
            if not os.path.exists(collection_path):
                return True
            
            # Check if any files have changed since last build
            meta_file_path = self.vector_store._get_meta_file_path(collection_name)
            if not os.path.exists(meta_file_path):
                return True
            
            # Compare current vs cached file states
            return not self.vector_store.is_collection_cache_valid(collection_name)
            
        except Exception:
            return True  # Err on the side of rebuilding
    
    def format_sources(self, source_chunks: List[Dict[str, Any]], format_type: str = "detailed") -> str:
        """
        Format source information for display
        
        Args:
            source_chunks: List of chunks that contributed to response
            format_type: "detailed", "compact", or "inline"
            
        Returns:
            Formatted source string
        """
        if not source_chunks:
            return ""
        
        if format_type == "compact":
            filenames = list(set(chunk["filename"] for chunk in source_chunks))
            return f"**Sources:** {', '.join(filenames)}"
        
        elif format_type == "inline":
            sources = []
            for i, chunk in enumerate(source_chunks, 1):
                sources.append(f"[{i}] {chunk['filename']}")
            return "\n".join(sources)
        
        else:  # detailed
            sources = ["**Sources:**"]
            for chunk in source_chunks:
                score_pct = int(chunk.get("similarity_score", 0) * 100)
                line_info = ""
                if chunk.get("start_line") and chunk.get("end_line"):
                    line_info = f" (lines {chunk['start_line']}-{chunk['end_line']})"
                
                sources.append(f"• {chunk['filename']}{line_info}, relevance: {score_pct}%")
            
            sources.append("\n- Use `--rag-show <filename>` to view relevant chunks")
            return "\n".join(sources)
    
    def show_chunk_in_file(self, filename: str) -> str:
        """Show relevant chunks within a file"""
        if not self.is_active():
            return "- No collection is active"
        
        # Find chunks from this file in the active collection
        if not self.active_collection_chunks:
            return "- No collection is active"
            
        file_chunks = [
            chunk for chunk in self.active_collection_chunks 
            if chunk["filename"] == filename
        ]
        
        if not file_chunks:
            return f"- File '{filename}' not found in collection '{self.active_collection}'"
        
        # Get the full file content
        if not self.active_collection:
            return "- No collection is active"
            
        collections_path = os.path.join(self.settings_manager.setting_get("working_dir"), "rag")
        collection_path = os.path.join(collections_path, self.active_collection)
        file_path = os.path.join(collection_path, filename)
        
        try:
            full_content = self.document_processor.load_file(file_path)
            if not full_content:
                return f"- Could not load file '{filename}'"
            
            # For now, just show the file content with chunk boundaries
            # TODO: Implement highlighting based on recent queries
            result = [f"- **{filename}** from collection '{self.active_collection}'"]
            result.append("=" * 50)
            result.append(full_content)
            
            return "\n".join(result)
            
        except Exception as e:
            return f"- Error reading file '{filename}': {e}"
    
    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a collection or the active collection"""
        target_collection = collection_name or self.active_collection
        
        if not target_collection:
            return {"error": "No collection specified and no active collection"}
        
        return self.vector_store.get_collection_info(target_collection)
    
    def is_active(self) -> bool:
        """Check if RAG is active with a collection loaded"""
        return self.active_collection is not None and self.active_collection_chunks is not None
    
    def get_active_collection_name(self) -> Optional[str]:
        """Get the name of the active collection"""
        return self.active_collection
    
    def get_status(self) -> Dict[str, Any]:
        """Get current RAG status"""
        status = {
            "active": self.is_active(),
            "active_collection": self.active_collection,
            "chunk_count": len(self.active_collection_chunks) if self.active_collection_chunks else 0,
            "available_collections": len(self.vector_store.get_available_collections()),
            "settings": {
                "embedding_model": self.settings_manager.setting_get("openai_embedding_model"),
                "chunk_size": self.settings_manager.setting_get("rag_chunk_size"),
                "chunk_overlap": self.settings_manager.setting_get("rag_chunk_overlap"),
                "top_k": self.settings_manager.setting_get("rag_top_k"),
                "context_strategy": self.settings_manager.setting_get("context_management_strategy")
            }
        }
        
        return status
    
    def refresh_collection(self, collection_name: str) -> bool:
        """Force refresh/rebuild a collection"""
        return self.build_collection(collection_name, force_rebuild=True)
    
    def delete_collection_cache(self, collection_name: str) -> bool:
        """Delete cached index for a collection"""
        success = self.vector_store.delete_collection_index(collection_name)
        
        # If this was the active collection, deactivate it
        if collection_name == self.active_collection:
            self.deactivate_collection()
        
        return success