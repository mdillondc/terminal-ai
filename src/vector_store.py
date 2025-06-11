import os
import json
import hashlib
import time
from typing import List, Dict, Any, Optional
from settings_manager import SettingsManager


class VectorStore:
    def __init__(self):
        self.settings_manager = SettingsManager.getInstance()
        self.working_dir = self.settings_manager.setting_get("working_dir")
        self.vectorstore_path = os.path.join(self.working_dir, "rag", "vectorstore")
        self.collections_path = os.path.join(self.working_dir, "rag")
        
        # Ensure vectorstore directory exists
        os.makedirs(self.vectorstore_path, exist_ok=True)
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _get_collection_file_metadata(self, collection_name: str) -> Dict[str, Dict[str, Any]]:
        """Get metadata for all files in a collection"""
        collection_path = os.path.join(self.collections_path, collection_name)
        if not os.path.exists(collection_path):
            return {}
        
        metadata = {}
        supported_extensions = {'.txt', '.md'}
        
        for filename in os.listdir(collection_path):
            file_path = os.path.join(collection_path, filename)
            
            if not os.path.isfile(file_path):
                continue
                
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in supported_extensions:
                continue
            
            try:
                stat = os.stat(file_path)
                metadata[filename] = {
                    "path": file_path,
                    "size": stat.st_size,
                    "modified_time": stat.st_mtime,
                    "hash": self._get_file_hash(file_path)
                }
            except Exception as e:
                print(f"Error getting metadata for {filename}: {e}")
                continue
        
        return metadata
    
    def _get_index_file_path(self, collection_name: str) -> str:
        """Get path to index file for a collection"""
        return os.path.join(self.vectorstore_path, f"{collection_name}_index.json")
    
    def _get_meta_file_path(self, collection_name: str) -> str:
        """Get path to metadata file for a collection"""
        return os.path.join(self.vectorstore_path, f"{collection_name}_meta.json")
    
    def save_collection_index(self, collection_name: str, chunks_with_embeddings: List[Dict[str, Any]]) -> bool:
        """
        Save collection index with embeddings and metadata
        
        Args:
            collection_name: Name of the collection
            chunks_with_embeddings: List of chunk dicts with 'embedding' field added
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            index_file_path = self._get_index_file_path(collection_name)
            meta_file_path = self._get_meta_file_path(collection_name)
            
            # Prepare index data
            index_data = {
                "collection_name": collection_name,
                "created_at": time.time(),
                "chunk_count": len(chunks_with_embeddings),
                "embedding_provider": self.settings_manager.setting_get("embedding_provider"),
                "embedding_model": self.settings_manager.setting_get("openai_embedding_model"),
                "ollama_embedding_model": self.settings_manager.setting_get("ollama_embedding_model"),
                "chunk_size": self.settings_manager.setting_get("rag_chunk_size"),
                "chunk_overlap": self.settings_manager.setting_get("rag_chunk_overlap"),
                "chunks": chunks_with_embeddings
            }
            
            # Save index file
            with open(index_file_path, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2, ensure_ascii=False)
            
            # Prepare and save metadata file
            file_metadata = self._get_collection_file_metadata(collection_name)
            meta_data = {
                "collection_name": collection_name,
                "index_created_at": time.time(),
                "file_metadata": file_metadata,
                "settings": {
                    "embedding_provider": self.settings_manager.setting_get("embedding_provider"),
                    "embedding_model": self.settings_manager.setting_get("openai_embedding_model"),
                    "ollama_embedding_model": self.settings_manager.setting_get("ollama_embedding_model"),
                    "chunk_size": self.settings_manager.setting_get("rag_chunk_size"),
                    "chunk_overlap": self.settings_manager.setting_get("rag_chunk_overlap")
                }
            }
            
            with open(meta_file_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)
            
            print(f"- Saved index for collection '{collection_name}' ({len(chunks_with_embeddings)} chunks)")
            return True
            
        except Exception as e:
            print(f"Error saving collection index for {collection_name}: {e}")
            return False
    
    def load_collection_index(self, collection_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load collection index with embeddings
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            List of chunks with embeddings, or None if not found/error
        """
        try:
            index_file_path = self._get_index_file_path(collection_name)
            
            if not os.path.exists(index_file_path):
                return None
            
            with open(index_file_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            chunks = index_data.get("chunks", [])
            print(f"- Loaded index for collection '{collection_name}' ({len(chunks)} chunks)")
            return chunks
            
        except Exception as e:
            print(f"Error loading collection index for {collection_name}: {e}")
            return None
    
    def is_collection_cache_valid(self, collection_name: str) -> bool:
        """
        Check if the cached index is still valid for a collection
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if cache is valid, False if needs rebuilding
        """
        try:
            meta_file_path = self._get_meta_file_path(collection_name)
            index_file_path = self._get_index_file_path(collection_name)
            
            # Check if both files exist
            if not (os.path.exists(meta_file_path) and os.path.exists(index_file_path)):
                return False
            
            # Load cached metadata
            with open(meta_file_path, 'r', encoding='utf-8') as f:
                cached_meta = json.load(f)
            
            # Get current file metadata
            current_meta = self._get_collection_file_metadata(collection_name)
            cached_file_meta = cached_meta.get("file_metadata", {})
            
            # Check if file count changed
            if len(current_meta) != len(cached_file_meta):
                return False
            
            # Check each file
            for filename, current_info in current_meta.items():
                if filename not in cached_file_meta:
                    return False
                
                cached_info = cached_file_meta[filename]
                
                # Check if file was modified
                if (current_info["modified_time"] != cached_info["modified_time"] or
                    current_info["size"] != cached_info["size"] or
                    current_info["hash"] != cached_info["hash"]):
                    return False
            
            # Check if settings changed
            current_settings = {
                "embedding_provider": self.settings_manager.setting_get("embedding_provider"),
                "embedding_model": self.settings_manager.setting_get("openai_embedding_model"),
                "ollama_embedding_model": self.settings_manager.setting_get("ollama_embedding_model"),
                "chunk_size": self.settings_manager.setting_get("rag_chunk_size"),
                "chunk_overlap": self.settings_manager.setting_get("rag_chunk_overlap")
            }
            cached_settings = cached_meta.get("settings", {})
            
            if current_settings != cached_settings:
                return False
            
            return True
            
        except Exception as e:
            print(f"Error checking cache validity for {collection_name}: {e}")
            return False
    
    def get_available_collections(self) -> List[str]:
        """Get list of available collection names"""
        try:
            if not os.path.exists(self.collections_path):
                return []
            
            collections = []
            for item in os.listdir(self.collections_path):
                item_path = os.path.join(self.collections_path, item)
                
                # Skip files and the vectorstore directory
                if not os.path.isdir(item_path) or item == "vectorstore":
                    continue
                
                # Check if directory has supported files
                has_supported_files = False
                supported_extensions = {'.txt', '.md'}
                
                try:
                    for filename in os.listdir(item_path):
                        if os.path.isfile(os.path.join(item_path, filename)):
                            file_ext = os.path.splitext(filename)[1].lower()
                            if file_ext in supported_extensions:
                                has_supported_files = True
                                break
                except:
                    continue
                
                if has_supported_files:
                    collections.append(item)
            
            return sorted(collections)
            
        except Exception as e:
            print(f"Error getting available collections: {e}")
            return []
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection"""
        try:
            collection_path = os.path.join(self.collections_path, collection_name)
            if not os.path.exists(collection_path):
                return {"error": "Collection not found"}
            
            # Get basic file info
            file_metadata = self._get_collection_file_metadata(collection_name)
            
            # Check if index exists and is valid
            has_index = os.path.exists(self._get_index_file_path(collection_name))
            cache_valid = self.is_collection_cache_valid(collection_name)
            
            # Get index info if available
            index_info = {}
            if has_index:
                try:
                    with open(self._get_index_file_path(collection_name), 'r') as f:
                        index_data = json.load(f)
                    index_info = {
                        "chunk_count": index_data.get("chunk_count", 0),
                        "created_at": index_data.get("created_at", 0),
                        "embedding_provider": index_data.get("embedding_provider", "unknown"),
                        "embedding_model": index_data.get("embedding_model", "unknown"),
                        "ollama_embedding_model": index_data.get("ollama_embedding_model", "unknown")
                    }
                except:
                    pass
            
            return {
                "name": collection_name,
                "path": collection_path,
                "file_count": len(file_metadata),
                "files": list(file_metadata.keys()),
                "has_index": has_index,
                "cache_valid": cache_valid,
                "index_info": index_info
            }
            
        except Exception as e:
            return {"error": f"Error getting collection info: {e}"}
    
    def delete_collection_index(self, collection_name: str) -> bool:
        """Delete index and metadata files for a collection"""
        try:
            index_file_path = self._get_index_file_path(collection_name)
            meta_file_path = self._get_meta_file_path(collection_name)
            
            deleted_files = []
            
            if os.path.exists(index_file_path):
                os.remove(index_file_path)
                deleted_files.append("index")
            
            if os.path.exists(meta_file_path):
                os.remove(meta_file_path)
                deleted_files.append("metadata")
            
            if deleted_files:
                print(f"- Deleted {', '.join(deleted_files)} for collection '{collection_name}'")
            
            return True
            
        except Exception as e:
            print(f"Error deleting collection index for {collection_name}: {e}")
            return False
    
    def get_vectorstore_stats(self) -> Dict[str, Any]:
        """Get statistics about the entire vector store"""
        try:
            stats = {
                "total_collections": 0,
                "collections_with_index": 0,
                "collections_with_valid_cache": 0,
                "total_chunks": 0,
                "vectorstore_size_mb": 0.0
            }
            
            collections = self.get_available_collections()
            stats["total_collections"] = len(collections)
            
            for collection_name in collections:
                info = self.get_collection_info(collection_name)
                
                if info.get("has_index"):
                    stats["collections_with_index"] += 1
                    stats["total_chunks"] += info.get("index_info", {}).get("chunk_count", 0)
                
                if info.get("cache_valid"):
                    stats["collections_with_valid_cache"] += 1
            
            # Calculate vectorstore directory size
            try:
                total_size = 0
                for filename in os.listdir(self.vectorstore_path):
                    file_path = os.path.join(self.vectorstore_path, filename)
                    if os.path.isfile(file_path):
                        total_size += os.path.getsize(file_path)
                stats["vectorstore_size_mb"] = round(total_size / (1024 * 1024), 2)
            except:
                pass
            
            return stats
            
        except Exception as e:
            return {"error": f"Error getting vectorstore stats: {e}"}