import pandas as pd
import networkx as nx
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('.'))

from src.network_builder import construct_partial_correlation_network, pivot_aspect_sentiments
from src.data_preprocessing import load_and_clean_data
from src.nlp_extraction import extract_aspects_and_sentiments

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def smoke_test():
    try:
        logger.info("Starting EBIC replacement smoke test...")
        
        # 1. Load small sample (1000 rows as requested)
        logger.info("Loading 1000 rows of data...")
        df = load_and_clean_data().head(1000)
        
        # 2. Extract aspects (minimal processing for speed if possible, but we need aspects)
        logger.info("Running NLP extraction on sample...")
        df = extract_aspects_and_sentiments(df)
        
        # 3. Pivot
        feature_matrix = pivot_aspect_sentiments(df)
        
        # 4. Run EBIC network construction
        logger.info("Running construct_partial_correlation_network (EBIC)...")
        G = construct_partial_correlation_network(feature_matrix)
        
        # 5. Verifications
        print("\n" + "="*30)
        print("SMOKE TEST RESULTS")
        print("="*30)
        
        # Check type
        is_graph = isinstance(G, nx.Graph)
        print(f"Returns NetworkX Graph: {is_graph}")
        
        # Check edges
        edge_count = G.number_of_edges()
        print(f"Number of edges returned: {edge_count}")
        
        if is_graph:
            print("SUCCESS: EBIC replacement is functional and compatible.")
        else:
            print("FAILURE: Function did not return a NetworkX Graph.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Smoke test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    smoke_test()
