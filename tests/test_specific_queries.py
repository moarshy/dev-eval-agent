#!/usr/bin/env python3

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def test_specific_queries():
    """Test specific queries on the vector store"""
    
    # Load the existing vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="tiangolo-fastapi"
    )
    
    # More specific test queries
    test_queries = [
        "FastAPI tutorial getting started",
        "request body",
        "path parameters",
        "middleware",
        "testing FastAPI"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print('='*60)
        try:
            results = vector_store.similarity_search(query, k=2)
            
            for i, result in enumerate(results, 1):
                metadata = result.metadata
                print(f"\n{i}. {metadata.get('section_title', 'No title')}")
                print(f"   Type: {metadata.get('content_type', 'unknown')}")
                print(f"   File: {metadata.get('file_path', 'N/A')}")
                print(f"   Content preview: {result.page_content[:200]}...")
                
        except Exception as e:
            print(f"Error querying '{query}': {e}")

if __name__ == "__main__":
    test_specific_queries()