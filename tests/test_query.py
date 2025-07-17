#!/usr/bin/env python3

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def test_vector_store_query():
    """Test querying the existing persisted vector store"""
    
    # Load the existing vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="tiangolo-fastapi"
    )
    
    # Test queries
    test_queries = [
        # "tit",
        # "title", 
        "authentication",
        # "how to create FastAPI app",
        # "dependency injection"
    ]
    
    for query in test_queries:
        print(f"\n=== Query: '{query}' ===")
        try:
            results = vector_store.similarity_search(query, k=3)
            print(f"Found {len(results)} results:")
            
            # for i, result in enumerate(results, 1):
            #     metadata = result.metadata
            #     print(f"\n{i}. {metadata.get('section_title', 'No title')} ({metadata.get('content_type', 'unknown')})")
            #     print(f"   Source: {metadata.get('file_path', 'N/A')}")
            #     print(f"   Content: {result.page_content[:150]}...")
                
            print(f"Results: {results}")
        except Exception as e:
            print(f"Error querying '{query}': {e}")

if __name__ == "__main__":
    test_vector_store_query()