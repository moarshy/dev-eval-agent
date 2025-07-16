#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'coding_agent_eval'))

from document_processor import VectorStoreManager
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def test_existing_vector_store():
    """Test querying the existing persisted vector store"""
    
    # Initialize the vector store manager with the same settings
    vector_manager = VectorStoreManager(
        persist_directory="./chroma_db",
        collection_name="tiangolo-fastapi"
    )
    
    # Load the existing vector store
    vector_manager.vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=vector_manager.embeddings,
        collection_name="tiangolo-fastapi"
    )
    
    # Test queries
    test_queries = [
        "tit",
        "title",
        "authentication",
        "how to create FastAPI app",
        "dependency injection"
    ]
    
    for query in test_queries:
        print(f"\n=== Query: '{query}' ===")
        try:
            results = vector_manager.similarity_search(query, k=3)
            print(f"Found {len(results)} results:")
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result.section_title} ({result.content_type})")
                print(f"   Source: {result.file_path or 'N/A'}")
                print(f"   Content: {result.content[:150]}...")
                
        except Exception as e:
            print(f"Error querying '{query}': {e}")

if __name__ == "__main__":
    test_existing_vector_store()