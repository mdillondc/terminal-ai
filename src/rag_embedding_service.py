import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
import ollama
import tiktoken
import requests
from settings_manager import SettingsManager
from rag_hybrid_search import HybridSearchService
from llm_client_manager import LLMClientManager
from print_helper import print_md


class EmbeddingService:
    def __init__(self, openai_client: OpenAI, ollama_client=None):
        self.openai_client = openai_client
        self.ollama_client = ollama_client
        self.settings_manager = SettingsManager.getInstance()
        self.encoding = tiktoken.get_encoding("cl100k_base")  # Standard encoding for embeddings

        # Create LLM client manager for hybrid search
        llm_client_manager = LLMClientManager(self.openai_client)
        self.hybrid_search = HybridSearchService(llm_client_manager)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))

    def _get_provider(self) -> str:
        """Get the current embedding provider from settings"""
        return self.settings_manager.setting_get("embedding_provider")

    def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI"""
        model = self.settings_manager.setting_get("openai_embedding_model")

        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            print_md(f"Error generating OpenAI embedding: {e}")
            raise

    def _generate_ollama_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama"""
        model = self.settings_manager.setting_get("ollama_embedding_model")

        try:
            if self.ollama_client is None:
                # Initialize ollama client if not provided
                ollama_url = self.settings_manager.setting_get("ollama_base_url")
                self.ollama_client = ollama.Client(host=ollama_url)

            response = self.ollama_client.embeddings(
                model=model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            print_md(f"Error generating Ollama embedding: {e}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using the configured provider"""
        if not text.strip():
            raise ValueError("Text cannot be empty")

        provider = self._get_provider()

        if provider == "ollama":
            return self._generate_ollama_embedding(text)
        elif provider == "openai":
            return self._generate_openai_embedding(text)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    def _generate_openai_embeddings_batch(self, texts: List[str], max_batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings using OpenAI in batches"""
        model = self.settings_manager.setting_get("openai_embedding_model")
        embeddings = []

        # Process in batches to avoid API limits
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i:i + max_batch_size]

            try:
                response = self.openai_client.embeddings.create(
                    input=batch,
                    model=model
                )

                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)

                # Small delay between batches to be respectful to API
                if i + max_batch_size < len(texts):
                    time.sleep(0.1)

            except Exception as e:
                print_md(f"Error generating OpenAI batch embeddings (batch {i//max_batch_size + 1}): {e}")
                raise

        return embeddings

    def _generate_ollama_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama with proper batching via /api/embed endpoint"""
        model = self.settings_manager.setting_get("ollama_embedding_model")
        ollama_url = self.settings_manager.setting_get("ollama_base_url")
        batch_size = self.settings_manager.setting_get("rag_batch_size")
        embeddings = []

        # Process texts in batches for optimal GPU utilization
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            try:
                # Use the new /api/embed endpoint with batch support
                response = requests.post(
                    f"{ollama_url}/api/embed",
                    json={
                        "model": model,
                        "input": batch_texts,
                        "keep_alive": "5m",  # Keep model loaded in GPU memory
                        "truncate": True
                    },
                    timeout=300  # 5 minutes timeout for large batches
                )

                if response.status_code != 200:
                    print_md(f"Ollama API error: {response.status_code} - {response.text}")
                    raise Exception(f"Ollama API request failed with status {response.status_code}")

                result = response.json()

                # Extract embeddings from response
                if 'embeddings' in result:
                    batch_embeddings = result['embeddings']
                    embeddings.extend(batch_embeddings)
                else:
                    print_md(f"Unexpected response format: {result}")
                    raise Exception("Invalid response format from Ollama API")

                # Progress tracking for large batches
                if len(texts) > 100:
                    processed = min(i + batch_size, len(texts))
                    print_md(f"Processed {processed}/{len(texts)} chunks...")

            except requests.exceptions.RequestException as e:
                print_md(f"Network error during batch embedding generation: {e}")
                # Fallback to individual processing for this batch
                batch_embeddings = self._generate_ollama_embeddings_fallback(batch_texts)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                print_md(f"Error generating Ollama batch embeddings: {e}")
                # Fallback to individual processing for this batch
                batch_embeddings = self._generate_ollama_embeddings_fallback(batch_texts)
                embeddings.extend(batch_embeddings)

        return embeddings

    def _generate_ollama_embeddings_fallback(self, texts: List[str]) -> List[List[float]]:
        """Fallback method for individual embedding generation when batch fails"""
        model = self.settings_manager.setting_get("ollama_embedding_model")
        embeddings = []

        if self.ollama_client is None:
            ollama_url = self.settings_manager.setting_get("ollama_base_url")
            self.ollama_client = ollama.Client(host=ollama_url)

        for text in texts:
            try:
                response = self.ollama_client.embeddings(
                    model=model,
                    prompt=text
                )
                embeddings.append(response['embedding'])
            except Exception as e:
                print_md(f"Error generating Ollama embedding for text: {e}")
                raise

        return embeddings

    def generate_embeddings_batch(self, texts: List[str], max_batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for multiple texts using the configured provider"""
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            raise ValueError("No valid texts provided")

        provider = self._get_provider()

        if provider == "ollama":
            return self._generate_ollama_embeddings_batch(valid_texts)
        elif provider == "openai":
            return self._generate_openai_embeddings_batch(valid_texts, max_batch_size)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    def cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same length")

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))

        # Calculate magnitudes
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0

        return dot_product / (magnitude1 * magnitude2)

    def find_most_similar(self, query_embedding: List[float],
                         document_embeddings: List[Dict[str, Any]],
                         top_k: Optional[int] = None,
                         query_text: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Find most similar document chunks to query embedding with hybrid search support

        Args:
            query_embedding: The query embedding vector
            document_embeddings: List of dicts with 'embedding' and metadata
            top_k: Number of top results to return (defaults to rag_top_k setting)
            query_text: Original query text for hybrid search

        Returns:
            List of document chunks sorted by hybrid score (highest first)
        """
        if top_k is None:
            top_k = self.settings_manager.setting_get("rag_top_k")

        # Ensure top_k is a valid integer
        if top_k is None or top_k <= 0:
            top_k = 5  # fallback default

        if not document_embeddings:
            return []

        # Calculate semantic similarities
        semantic_scores = []
        valid_chunks = []

        for doc_chunk in document_embeddings:
            if 'embedding' not in doc_chunk:
                continue

            similarity = self.cosine_similarity(query_embedding, doc_chunk['embedding'])
            semantic_scores.append(similarity)
            valid_chunks.append(doc_chunk)

        if not valid_chunks:
            return []

        # Use simplified search if enabled and query text is provided
        if self.hybrid_search.enabled and query_text:
            return self.hybrid_search.search(
                query_text, valid_chunks, semantic_scores, top_k
            )
        else:
            # Fall back to pure semantic search with diversity
            results = []
            for i, chunk in enumerate(valid_chunks):
                result = chunk.copy()
                result['similarity_score'] = semantic_scores[i]
                result['final_score'] = semantic_scores[i]
                results.append(result)

            # Sort by similarity (highest first)
            results.sort(key=lambda x: x['final_score'], reverse=True)

            # Apply result diversity if enabled
            if self.settings_manager.setting_get("rag_enable_result_diversity"):
                results = self._apply_semantic_result_diversity(results, top_k)
            else:
                results = results[:top_k]

            return results

    def _apply_semantic_result_diversity(self, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Apply result diversity to semantic search results

        Args:
            results: Sorted list of semantic search results
            top_k: Maximum number of results to return

        Returns:
            Diversified list of results
        """
        if not results:
            return results

        max_chunks_per_source = self.settings_manager.setting_get("rag_max_chunks_per_source")

        # Group results by source filename
        source_counts = {}
        diversified_results = []

        for result in results:
            source_filename = result.get('filename', 'unknown')

            # Count how many chunks we've already taken from this source
            current_count = source_counts.get(source_filename, 0)

            if current_count < max_chunks_per_source:
                diversified_results.append(result)
                source_counts[source_filename] = current_count + 1

                # Stop if we've reached the desired number of results
                if len(diversified_results) >= top_k:
                    break

        return diversified_results

    def get_embedding_dimensions(self) -> int:
        """Get the embedding dimensions for the current provider and model"""
        provider = self._get_provider()

        if provider == "openai":
            model = self.settings_manager.setting_get("openai_embedding_model")
            openai_dimensions = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            return openai_dimensions.get(model, 1536)
        elif provider == "ollama":
            model = self.settings_manager.setting_get("ollama_embedding_model")
            ollama_dimensions = {
                "nomic-embed-text": 768,
                "all-minilm": 384,
                "mxbai-embed-large": 1024,
                "xlm-roberta-large-passage-rerank": 1024,
                "snowflake-arctic-embed2:latest": 1024
            }
            return ollama_dimensions.get(model, 768)
        else:
            return 768  # Default fallback

    def get_embedding_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model"""
        provider = self._get_provider()

        if provider == "openai":
            model = self.settings_manager.setting_get("openai_embedding_model")

            # OpenAI model information
            openai_model_info = {
                "text-embedding-3-small": {
                    "dimensions": 1536,
                    "max_tokens": 8191,
                    "cost_per_1k_tokens": 0.00002
                },
                "text-embedding-3-large": {
                    "dimensions": 3072,
                    "max_tokens": 8191,
                    "cost_per_1k_tokens": 0.00013
                },
                "text-embedding-ada-002": {
                    "dimensions": 1536,
                    "max_tokens": 8191,
                    "cost_per_1k_tokens": 0.0001
                }
            }

            return {
                "provider": "openai",
                "model": model,
                "info": openai_model_info.get(model, {"dimensions": "unknown", "max_tokens": 8191})
            }

        elif provider == "ollama":
            model = self.settings_manager.setting_get("ollama_embedding_model")

            # Ollama model information
            ollama_model_info = {
                "nomic-embed-text": {
                    "dimensions": 768,
                    "max_tokens": 8192,
                    "cost_per_1k_tokens": 0.0,
                    "multilingual": True,
                    "languages": "Multi-lingual"
                },

                "snowflake-arctic-embed2:latest": {
                    "dimensions": 1024,
                    "max_tokens": 8192,
                    "cost_per_1k_tokens": 0.0,
                    "multilingual": True,
                    "languages": "Multi-lingual"
                }
            }

            return {
                "provider": "ollama",
                "model": model,
                "info": ollama_model_info.get(model, {"dimensions": 768, "max_tokens": 8192})
            }

        else:
            return {
                "provider": "unknown",
                "model": "unknown",
                "info": {"dimensions": "unknown"}
            }

    def test_connection(self) -> bool:
        """Test connection to the current embedding provider"""
        provider = self._get_provider()

        try:
            # Test with a simple text
            test_text = "Hello, this is a test."
            self.generate_embedding(test_text)
            return True
        except Exception as e:
            print_md(f"Connection test failed for {provider}: {e}")
            return False