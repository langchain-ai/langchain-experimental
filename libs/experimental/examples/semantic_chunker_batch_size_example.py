"""Example demonstrating the use of embedding_batch_size parameter in SemanticChunker.

This example shows how to use the embedding_batch_size parameter to handle:
1. API batch size limits (e.g., some APIs like Qwen3 have batch size restrictions)
2. Memory constraints when processing large documents
"""

from typing import List

from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker


class MockEmbeddings(Embeddings):
    """Mock embeddings for demonstration purposes."""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        print(f"Embedding batch of {len(texts)} documents")
        # In real scenarios, this would call an actual embedding API
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return [0.1, 0.2, 0.3]


def main() -> None:
    """Demonstrate the embedding_batch_size parameter."""
    
    # Sample text to split
    text = """
    Machine learning is a subset of artificial intelligence. It focuses on the 
    development of algorithms that can learn from data. Deep learning is a 
    specialized form of machine learning. It uses neural networks with multiple 
    layers. Natural language processing is another important AI field. It deals 
    with the interaction between computers and human language. Computer vision 
    enables machines to interpret visual information. It has applications in 
    autonomous vehicles and medical imaging. Reinforcement learning teaches 
    agents through trial and error. It has been successful in game playing and 
    robotics.
    """
    
    print("=" * 80)
    print("Example 1: Without batch_size (default behavior)")
    print("=" * 80)
    
    embeddings = MockEmbeddings()
    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
    )
    
    chunks = chunker.split_text(text)
    print(f"\nGenerated {len(chunks)} chunks")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
    
    print("\n" + "=" * 80)
    print("Example 2: With embedding_batch_size=2")
    print("=" * 80)
    print("(Notice how embeddings are processed in smaller batches)")
    
    embeddings_batched = MockEmbeddings()
    chunker_batched = SemanticChunker(
        embeddings=embeddings_batched,
        breakpoint_threshold_type="percentile",
        embedding_batch_size=2,  # Process embeddings in batches of 2
    )
    
    chunks_batched = chunker_batched.split_text(text)
    print(f"\nGenerated {len(chunks_batched)} chunks")
    for i, chunk in enumerate(chunks_batched, 1):
        print(f"\nChunk {i}:")
        print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
    
    print("\n" + "=" * 80)
    print("Key Benefits:")
    print("=" * 80)
    print("1. Respects API batch size limits")
    print("2. Reduces memory usage for large documents")
    print("3. Maintains same chunking quality")
    print("4. Provides better control over embedding operations")


if __name__ == "__main__":
    main()
