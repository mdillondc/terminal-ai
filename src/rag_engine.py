import os
import time
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import ollama
from document_processor import DocumentProcessor
from rag_embedding_service import EmbeddingService
from vector_store import VectorStore
from settings_manager import SettingsManager
from print_helper import print_md

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

        # Apply current embedding profile to vector store
        try:
            profile = self.embedding_service.get_current_embedding_profile()
            self.vector_store.set_embedding_profile(profile.get("provider"), profile.get("model"))
        except Exception:
            pass

        # Track active embedding profile for reloading on provider/model change
        self.active_profile_provider = None
        self.active_profile_model = None

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

    def build_collection(self, collection_name: str, force_rebuild: bool = False, force_full: bool = False) -> bool:
        """
        Build/rebuild embeddings for a collection

        Args:
            collection_name: Name of the collection to build
            force_rebuild: If True, rebuild even if cache is valid
            force_full: If True, force full rebuild (ignore smart rebuild)

        Returns:
            True if successful, False otherwise
        """
        if not self.collection_exists(collection_name):
            print_md(f"Collection '{collection_name}' not found")
            return False

        # Check if rebuild is needed
        if not force_rebuild and self.vector_store.is_collection_cache_valid(collection_name):
            print_md(f"Collection '{collection_name}' index is up to date")
            return True

        # Choose rebuild strategy
        if force_full:
            return self._build_collection_full(collection_name)
        else:
            return self._build_collection_smart(collection_name)

    def _build_collection_full(self, collection_name: str) -> bool:
        """
        Full rebuild of collection - processes all files from scratch

        Args:
            collection_name: Name of the collection to build

        Returns:
            True if successful, False otherwise
        """
        print_md(f"Full rebuild of collection '{collection_name}'...")

        try:
            # Get collection path
            collections_path = os.path.join(self.settings_manager.setting_get("working_dir"), "rag")
            collection_path = os.path.join(collections_path, collection_name)

            # Process documents into chunks
            print_md("Processing documents...")
            chunks = self.document_processor.process_collection(collection_path)

            if not chunks:
                print_md("No processable documents found in collection")
                return False

            # Generate embeddings
            print_md(f"Generating embeddings for {len(chunks)} chunks...")
            chunk_texts = [chunk["content"] for chunk in chunks]

            embeddings = self.embedding_service.generate_embeddings_batch(chunk_texts)

            if len(embeddings) != len(chunks):
                print_md("Mismatch between chunks and embeddings")
                return False

            # Add embeddings to chunks
            for i, embedding in enumerate(embeddings):
                chunks[i]["embedding"] = embedding
                chunks[i]["created_at"] = time.time()

            # Save to vector store
            print_md("Saving index...")
            success = self.vector_store.save_collection_index(collection_name, chunks)

            if success:
                print_md(f"Successfully built collection '{collection_name}'")

                # If this is the active collection, reload it
                if self.active_collection == collection_name:
                    self.active_collection_chunks = chunks

                return True
            else:
                print_md(f"Failed to save collection '{collection_name}'")
                return False

        except Exception as e:
            print_md(f"Error building collection '{collection_name}': {e}")
            return False

    def _build_collection_smart(self, collection_name: str) -> bool:
        """
        Smart rebuild of collection - only processes changed files

        Args:
            collection_name: Name of the collection to build

        Returns:
            True if successful, False otherwise
        """
        print_md(f"Smart rebuild of collection '{collection_name}'...")

        try:
            # Get collection path
            collections_path = os.path.join(self.settings_manager.setting_get("working_dir"), "rag")
            collection_path = os.path.join(collections_path, collection_name)

            # Get file changes
            changed_files, unchanged_files, deleted_files = self.vector_store.get_file_changes(collection_name)

            print_md(f"Found {len(changed_files)} changed files, {len(unchanged_files)} unchanged files, {len(deleted_files)} deleted files")

            # If no existing index or all files changed, do full rebuild
            if not unchanged_files and not os.path.exists(self.vector_store._get_index_file_path(collection_name)):
                print_md("No existing index found, performing full rebuild...")
                return self._build_collection_full(collection_name)

            # Load existing chunks for unchanged files
            existing_chunks = []
            if unchanged_files:
                print_md(f"Loading existing chunks for {len(unchanged_files)} unchanged files...")
                existing_chunks = self.vector_store.load_chunks_for_files(collection_name, unchanged_files)

            # Process changed files
            new_chunks = []
            if changed_files:
                print_md(f"Processing {len(changed_files)} changed files...")
                for file_path in changed_files:
                    full_file_path = os.path.join(collection_path, file_path)
                    if os.path.exists(full_file_path):
                        try:
                            file_chunks = self.document_processor.process_file(full_file_path, collection_name, collection_path)
                            new_chunks.extend(file_chunks)
                        except Exception as e:
                            print_md(f"Error processing {file_path}: {e}")
                            continue

            # Generate embeddings for new chunks only
            if new_chunks:
                print_md(f"Generating embeddings for {len(new_chunks)} new chunks...")
                chunk_texts = [chunk["content"] for chunk in new_chunks]
                embeddings = self.embedding_service.generate_embeddings_batch(chunk_texts)

                if len(embeddings) != len(new_chunks):
                    print_md("Mismatch between new chunks and embeddings")
                    return False

                # Add embeddings to new chunks
                for i, embedding in enumerate(embeddings):
                    new_chunks[i]["embedding"] = embedding
                    new_chunks[i]["created_at"] = time.time()

            # Combine existing and new chunks
            all_chunks = existing_chunks + new_chunks

            if not all_chunks:
                print_md("No processable documents found in collection")
                return False

            # Save to vector store
            print_md("Saving index...")
            success = self.vector_store.save_collection_index(collection_name, all_chunks)

            if success:
                total_chunks = len(all_chunks)
                new_chunk_count = len(new_chunks)
                reused_chunk_count = len(existing_chunks)
                print_md(f"Successfully built collection '{collection_name}' ({total_chunks} chunks: {new_chunk_count} new, {reused_chunk_count} reused)")

                # If this is the active collection, reload it
                if self.active_collection == collection_name:
                    self.active_collection_chunks = all_chunks

                return True
            else:
                print_md(f"Failed to save collection '{collection_name}'")
                return False

        except Exception as e:
            print_md(f"Error smart building collection '{collection_name}': {e}")
            # On error, fall back to full rebuild
            print_md("Falling back to full rebuild...")
            return self._build_collection_full(collection_name)

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
                print_md(f"Collection '{collection_name}' not found")
            return False

        # Check if index exists
        # Ensure vector store points at current embedding profile
        profile = self.embedding_service.get_current_embedding_profile()
        self.vector_store.set_embedding_profile(profile.get("provider"), profile.get("model"))

        index_exists = os.path.exists(self.vector_store._get_index_file_path(collection_name))
        cache_valid = self.vector_store.is_collection_cache_valid(collection_name)

        if not index_exists:
            if verbose:
                print_md(f"Collection '{collection_name}' has no index")
                print_md(f"Auto-building collection...")

            # Auto-build the collection
            success = self.build_collection(collection_name, force_rebuild=True, force_full=False)
            if not success:
                if verbose:
                    print_md(f"Failed to build collection '{collection_name}'")
                return False

        if not cache_valid:
            if verbose:
                print_md(f"Collection '{collection_name}' files have changed since last build")
                print_md(f"Auto-rebuilding collection...")

            # Auto-rebuild the collection
            success = self.build_collection(collection_name, force_rebuild=True, force_full=False)
            if not success:
                if verbose:
                    print_md(f"Failed to rebuild collection '{collection_name}'")
                return False

        # Load collection index
        chunks = self.vector_store.load_collection_index(collection_name)
        if chunks is None:
            if verbose:
                print_md(f"Failed to load collection '{collection_name}'")
            return False

        # Activate collection
        self.active_collection = collection_name
        self.active_collection_chunks = chunks

        # Record the active embedding profile for this loaded collection
        try:
            profile = self.embedding_service.get_current_embedding_profile()
            self.active_profile_provider = profile.get("provider")
            self.active_profile_model = profile.get("model")
        except Exception:
            self.active_profile_provider = None
            self.active_profile_model = None

        # Save to settings
        self.settings_manager.setting_set("rag_active_collection", collection_name)

        if verbose:
            print_md(f"Activated collection '{collection_name}' ({len(chunks)} chunks)")

        return True

    def deactivate_collection(self) -> None:
        """Deactivate the current collection"""
        if self.active_collection:
            print_md(f"Deactivated collection '{self.active_collection}'")

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
            # Ensure vector store uses current embedding profile
            profile = self.embedding_service.get_current_embedding_profile()
            self.vector_store.set_embedding_profile(profile.get("provider"), profile.get("model"))

            # If embedding profile changed since activation, reload the collection for the current profile
            if (self.active_profile_provider and self.active_profile_model) and (
                self.active_profile_provider != profile.get("provider") or self.active_profile_model != profile.get("model")
            ):
                print_md("Embedding provider/model changed since activation; reloading collection for current profile...")
                if not self.activate_collection(self.active_collection, verbose=True):
                    return []

            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query_text)

            # Find similar chunks, passing original query for hybrid search
            results = self.embedding_service.find_most_similar(
                query_embedding,
                self.active_collection_chunks,
                top_k,
                query_text  # Pass original query for hybrid search analysis
            )

            return results

        except ValueError as e:
            # Provide clearer guidance when dimension mismatch occurs
            if "same length" in str(e).lower():
                print_md("Error querying collection: Embedding dimension mismatch detected. This usually happens when the index was built with a different embedding provider/model. Rebuild the collection for the current provider/model, or activate a profile-specific index that matches the current provider.")
                return []
            else:
                print_md(f"Error querying collection: {e}")
                return []
        except Exception as e:
            print_md(f"Error querying collection: {e}")
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

        # Format context
        context_parts = []
        used_chunks = []

        context_header = f"You have access to relevant information from the user's document collection '{self.active_collection}':\n\nCONTEXT:\n---\n"

        for chunk in relevant_chunks:
            chunk_text = f"[From {chunk['filename']}]\n{chunk['content']}\n\n"
            context_parts.append(chunk_text)
            used_chunks.append(chunk)

        if not context_parts:
            return "", []

        context_footer = "---\n\nUse this context to enhance your responses. Cite sources when referencing specific information."

        full_context = context_header + "".join(context_parts) + context_footer

        return full_context, used_chunks


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

                sources.append(f"    {chunk['filename']}{line_info}, relevance: {score_pct}%")

            sources.append("Use `--rag-show <filename>` to view relevant chunks\n")
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
            result.append("\n")
            result.append("=" * 50)
            result.append(full_content)
            result.append("=" * 50)

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
                "embedding_provider": self.embedding_service.get_current_embedding_profile().get("provider"),
                "embedding_model": self.embedding_service.get_current_embedding_profile().get("model"),
                "embedding_dimensions": self.embedding_service.get_current_embedding_profile().get("dimensions"),
                "chunk_size": self.settings_manager.setting_get("rag_chunk_size"),
                "chunk_overlap": self.settings_manager.setting_get("rag_chunk_overlap"),
                "top_k": self.settings_manager.setting_get("rag_top_k")
            }
        }

        return status

    def delete_collection_cache(self, collection_name: str) -> bool:
        """Delete cached index for a collection"""
        success = self.vector_store.delete_collection_index(collection_name)

        # If this was the active collection, deactivate it
        if collection_name == self.active_collection:
            self.deactivate_collection()

        return success