import os
import hashlib
import time
import pandas as pd
import numpy as np
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
        supported_extensions = {'.txt', '.md', '.pdf'}
        
        for root, dirs, files in os.walk(collection_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext not in supported_extensions:
                    continue
                
                # Get relative path for better organization
                relative_path = os.path.relpath(file_path, collection_path)
                
                try:
                    stat = os.stat(file_path)
                    metadata[relative_path] = {
                        "path": file_path,
                        "size": stat.st_size,
                        "modified_time": stat.st_mtime,
                        "hash": self._get_file_hash(file_path)
                    }
                except Exception as e:
                    print(f"Error getting metadata for {relative_path}: {e}")
                    continue
        
        return metadata
    
    def _get_index_file_path(self, collection_name: str) -> str:
        """Get path to index file for a collection"""
        return os.path.join(self.vectorstore_path, f"{collection_name}_index.parquet")
    
    def _get_meta_file_path(self, collection_name: str) -> str:
        """Get path to metadata file for a collection"""
        return os.path.join(self.vectorstore_path, f"{collection_name}_meta.parquet")
    
    def save_collection_index(self, collection_name: str, chunks_with_embeddings: List[Dict[str, Any]]) -> bool:
        """
        Save collection index with embeddings and metadata using Parquet format
        
        Args:
            collection_name: Name of the collection
            chunks_with_embeddings: List of chunk dicts with 'embedding' field added
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            index_file_path = self._get_index_file_path(collection_name)
            meta_file_path = self._get_meta_file_path(collection_name)
            
            # Prepare chunks data for Parquet storage
            chunks_data = []
            for chunk in chunks_with_embeddings:
                chunk_copy = chunk.copy()
                # Convert embedding to numpy array for efficient storage
                if 'embedding' in chunk_copy:
                    chunk_copy['embedding'] = np.array(chunk_copy['embedding'], dtype=np.float32)
                chunks_data.append(chunk_copy)
            
            # Create DataFrame and save as Parquet
            chunks_df = pd.DataFrame(chunks_data)
            chunks_df.to_parquet(index_file_path, compression='snappy', index=False)
            
            # Prepare and save metadata
            file_metadata = self._get_collection_file_metadata(collection_name)
            meta_data = {
                "collection_name": collection_name,
                "index_created_at": time.time(),
                "chunk_count": len(chunks_with_embeddings),
                "embedding_provider": self.settings_manager.setting_get("embedding_provider"),
                "embedding_model": self.settings_manager.setting_get("openai_embedding_model"),
                "ollama_embedding_model": self.settings_manager.setting_get("ollama_embedding_model"),
                "chunk_size": self.settings_manager.setting_get("rag_chunk_size"),
                "chunk_overlap": self.settings_manager.setting_get("rag_chunk_overlap"),
            }
            
            # Convert file metadata to DataFrame format
            meta_rows = []
            for filename, file_info in file_metadata.items():
                meta_rows.append({
                    "filename": filename,
                    "file_path": file_info["path"],
                    "file_size": file_info["size"],
                    "modified_time": file_info["modified_time"],
                    "file_hash": file_info["hash"]
                })
            
            # Create metadata DataFrame with collection info
            if meta_rows:
                file_meta_df = pd.DataFrame(meta_rows)
            else:
                file_meta_df = pd.DataFrame(columns=["filename", "file_path", "file_size", "modified_time", "file_hash"])
            
            # Add collection metadata as attributes (we'll store them separately)
            collection_meta_df = pd.DataFrame([meta_data])
            
            # Save both as a single Parquet file with multiple row groups
            # We'll use the first row for collection metadata and the rest for file metadata
            combined_meta = pd.concat([collection_meta_df, file_meta_df], ignore_index=True, sort=False)
            combined_meta.to_parquet(meta_file_path, compression='snappy', index=False)
            
            print(f"- Saved index for collection '{collection_name}' ({len(chunks_with_embeddings)} chunks)")
            return True
            
        except Exception as e:
            print(f"Error saving collection index for {collection_name}: {e}")
            return False
    
    def load_collection_index(self, collection_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load collection index with embeddings from Parquet format
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            List of chunks with embeddings, or None if not found/error
        """
        try:
            index_file_path = self._get_index_file_path(collection_name)
            
            if not os.path.exists(index_file_path):
                return None
            
            # Load chunks from Parquet
            chunks_df = pd.read_parquet(index_file_path)
            
            # Convert back to list of dictionaries
            chunks = chunks_df.to_dict('records')
            
            # Convert numpy arrays back to lists for compatibility
            for chunk in chunks:
                if 'embedding' in chunk and isinstance(chunk['embedding'], np.ndarray):
                    chunk['embedding'] = chunk['embedding'].tolist()
            
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
            meta_df = pd.read_parquet(meta_file_path)
            
            # First row contains collection metadata, rest contains file metadata
            if len(meta_df) == 0:
                return False
            
            collection_meta = meta_df.iloc[0].to_dict()
            
            # Get current file metadata
            current_meta = self._get_collection_file_metadata(collection_name)
            
            # Extract file metadata from the DataFrame (skip first row which is collection metadata)
            cached_file_meta = {}
            if len(meta_df) > 1:
                file_rows = meta_df.iloc[1:].copy()
                file_rows = file_rows.dropna(subset=['filename'])  # Remove rows without filename
                
                for _, row in file_rows.iterrows():
                    if pd.notna(row['filename']):
                        cached_file_meta[row['filename']] = {
                            "path": row['file_path'],
                            "size": row['file_size'],
                            "modified_time": row['modified_time'],
                            "hash": row['file_hash']
                        }
            
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
            
            # Compare with cached settings from collection metadata
            cached_settings = {
                "embedding_provider": collection_meta.get("embedding_provider"),
                "embedding_model": collection_meta.get("embedding_model"),
                "ollama_embedding_model": collection_meta.get("ollama_embedding_model"),
                "chunk_size": collection_meta.get("chunk_size"),
                "chunk_overlap": collection_meta.get("chunk_overlap")
            }
            
            if current_settings != cached_settings:
                return False
            
            return True
            
        except Exception as e:
            print(f"Error checking cache validity for {collection_name}: {e}")
            return False
    
    def get_available_collections(self) -> List[str]:
        """Get list of available collection names - just show all directories in rag/ except vectorstore"""
        try:
            if not os.path.exists(self.collections_path):
                return []
            
            collections = []
            for item in os.listdir(self.collections_path):
                item_path = os.path.join(self.collections_path, item)
                
                # Only include directories, skip the vectorstore directory
                if os.path.isdir(item_path) and item != "vectorstore":
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
                    meta_df = pd.read_parquet(self._get_meta_file_path(collection_name))
                    if len(meta_df) > 0:
                        collection_meta = meta_df.iloc[0].to_dict()
                        index_info = {
                            "chunk_count": int(collection_meta.get("chunk_count", 0)) if pd.notna(collection_meta.get("chunk_count")) else 0,
                            "created_at": float(collection_meta.get("index_created_at", 0)) if pd.notna(collection_meta.get("index_created_at")) else 0,
                            "embedding_provider": str(collection_meta.get("embedding_provider", "unknown")),
                            "embedding_model": str(collection_meta.get("embedding_model", "unknown")),
                            "ollama_embedding_model": str(collection_meta.get("ollama_embedding_model", "unknown"))
                        }
                except Exception as e:
                    print(f"Error reading index info: {e}")
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