"""
Qdrant Collections Checker
==========================
Connects to Qdrant Cloud and displays information about existing collections,
including collection names, vector counts, and vector dimensions.

Usage:
    python check_collections.py
"""

import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def check_collections(qdrant_url=None, qdrant_api_key=None):
    """
    Connect to Qdrant Cloud and display collection information.
    
    Args:
        qdrant_url: Qdrant Cloud URL (default: from QDRANT_URL_v2 env var)
        qdrant_api_key: Qdrant API key (default: from QDRANT_API_KEY_v2 env var)
    """
    # Load environment variables
    load_dotenv()
    
    qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
    qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url or not qdrant_api_key:
        print("[ERROR] Missing Qdrant credentials")
        print("Ensure QDRANT_URL and QDRANT_API_KEY are set in .env file")
        return False
    
    try:
        # Connect to Qdrant Cloud
        print(f"\n{'='*70}")
        print("QDRANT COLLECTIONS CHECKER")
        print(f"{'='*70}\n")
        
        print("[INFO] Connecting to Qdrant Cloud...")
        print(f"  URL: {qdrant_url}")
        
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        print("[OK] Connected successfully\n")
        
        # Get collections
        print("[INFO] Fetching collections...\n")
        collections_response = client.get_collections()
        collections = collections_response.collections
        
        if not collections:
            print("[INFO] No collections found in Qdrant")
            return True
        
        print(f"{'='*70}")
        print(f"FOUND {len(collections)} COLLECTION(S)")
        print(f"{'='*70}\n")
        
        # Display each collection's information
        for i, collection in enumerate(collections, 1):
            print(f"[{i}] Collection Name: {collection.name}")
            
            # Try to get detailed collection info
            try:
                collection_info = client.get_collection(collection.name)
                
                # Display points count if available
                if hasattr(collection_info, 'points_count'):
                    print(f"    Points Count: {collection_info.points_count}")
                elif hasattr(collection, 'points_count'):
                    print(f"    Points Count: {collection.points_count}")
                
                # Display status if available
                if hasattr(collection_info, 'status'):
                    print(f"    Status: {collection_info.status}")
                
                # Display vector configuration
                if hasattr(collection_info, 'config') and collection_info.config:
                    if hasattr(collection_info.config, 'params') and collection_info.config.params:
                        params = collection_info.config.params
                        
                        if hasattr(params, 'vectors') and params.vectors:
                            vectors_config = params.vectors
                            
                            # Check if it's a named vectors config (dict) or single vector
                            if isinstance(vectors_config, dict):
                                print(f"    Vector Spaces (Named Vectors):")
                                for vector_name, vector_params in vectors_config.items():
                                    dist_metric = getattr(vector_params, 'distance', 'Unknown')
                                    print(f"      - {vector_name}: {vector_params.size}D ({dist_metric})")
                            else:
                                # Single vector config
                                dist_metric = getattr(vectors_config, 'distance', 'Unknown')
                                print(f"    Vector Dimension: {vectors_config.size}D")
                                print(f"    Distance Metric: {dist_metric}")
                
                # Display full configuration if available
                if hasattr(collection_info, 'config'):
                    print(f"    Configuration: {collection_info.config}")
                
            except Exception as e:
                print(f"    [WARN] Could not fetch detailed info: {e}")
                # Still try to show basic info from collection object
                if hasattr(collection, 'points_count'):
                    print(f"    Points Count: {collection.points_count}")
            
            print()
        
        # Print summary
        print(f"{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Total Collections: {len(collections)}")
        
        # Try to calculate total points
        total_points = 0
        for col in collections:
            try:
                collection_info = client.get_collection(col.name)
                if hasattr(collection_info, 'points_count'):
                    total_points += collection_info.points_count
                elif hasattr(col, 'points_count'):
                    total_points += col.points_count
            except:
                pass
        
        print(f"Total Points: {total_points}")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to check collections: {e}")
        print("\nTroubleshooting:")
        print("  1. Check that QDRANT_URL and QDRANT_API_KEY are set in .env")
        print("  2. Verify your Qdrant Cloud URL is accessible")
        print("  3. Verify your API key is valid")
        return False


def main():
    """Main entry point."""
    success = check_collections()
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
