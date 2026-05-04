import pandas as pd
import logging
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_clean_data() -> pd.DataFrame:
    """
    Loads the 'All_Beauty' category from the Amazon Reviews 2023 dataset on Hugging Face 
    and performs basic cleaning.

    The dataset is loaded directly into memory without local storage. 
    It filters for specific columns and removes rows with missing values 
    in critical fields.

    Returns:
        pd.DataFrame: A cleaned DataFrame containing columns: 
                      'rating', 'title', 'text', 'user_id', 'parent_asin'.
    """
    logger.info("Starting to load dataset from Hugging Face: McAuley-Lab/Amazon-Reviews-2023 (raw_review_All_Beauty)")
    
    try:
        # Load the dataset directly into memory
        # Using 'full' split as requested (alternatively 'train' if 'full' is not available)
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023", 
            "raw_review_All_Beauty", 
            split="full", 
            trust_remote_code=True
        )
    except ValueError:
        logger.warning("'full' split not found, attempting to load 'train' split.")
        dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023", 
            "raw_review_All_Beauty", 
            split="train", 
            trust_remote_code=True
        )
    
    # Convert to pandas DataFrame
    df = dataset.to_pandas()
    logger.info(f"Dataset loaded. Initial shape: {df.shape}")
    
    # Filter columns
    essential_columns = ['rating', 'title', 'text', 'user_id', 'parent_asin']
    # Check if columns exist before filtering to avoid KeyErrors
    df = df[[col for col in essential_columns if col in df.columns]]
    logger.info(f"Filtered columns. Current columns: {df.columns.tolist()}")
    
    # Drop rows with null values in 'text' or 'rating'
    initial_len = len(df)
    df = df.dropna(subset=['text', 'rating'])
    final_len = len(df)
    
    logger.info(f"Dropped {initial_len - final_len} rows with missing 'text' or 'rating'.")
    logger.info(f"Final DataFrame shape: {df.shape}")
    
    return df

if __name__ == "__main__":
    try:
        data = load_and_clean_data()
        print("\n--- DataFrame Info ---")
        print(data.info())
        print("\n--- First 5 Rows ---")
        print(data.head())
    except Exception as e:
        logger.error(f"An error occurred during standalone execution: {e}")
