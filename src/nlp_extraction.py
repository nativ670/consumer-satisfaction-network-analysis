import pandas as pd
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import logging
from tqdm import tqdm
import torch
from src.data_preprocessing import load_and_clean_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Predefined core beauty aspects for categorization and noise reduction
CORE_ASPECTS = [
    'smell/fragrance', 
    'price/value', 
    'texture/consistency', 
    'packaging', 
    'ingredients', 
    'effectiveness/results', 
    'service/shipping'
]

def extract_aspects_and_sentiments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs Aspect-Based Sentiment Analysis (ABSA) on the 'text' column of the DataFrame.
    
    1. Extracts raw aspects (noun chunks) using spaCy.
    2. Maps raw aspects to core beauty aspects using Sentence-Transformers (MiniLM).
    3. Filters out noise based on a semantic similarity threshold.
    4. Calculates context-aware sentiment scores for each matched aspect using a 
       dedicated DeBERTa-based ABSA model.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing at least a 'text' column.
        
    Returns:
        pd.DataFrame: DataFrame with an additional 'aspect_sentiments' column.
                      Each entry is a list of tuples: [(core_aspect, sentiment_score), ...].
    """
    logger.info("Initializing NLP models (spaCy, Sentence-Transformers, and DeBERTa ABSA)...")
    
    # Load spaCy model for raw aspect (noun chunk) extraction
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading spaCy model 'en_core_web_sm'...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Determine execution device
    device_idx = 0 if torch.cuda.is_available() else -1
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Similarity model for aspect categorization
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=device_str)
    # Use space-separated core aspects for better semantic matching
    core_aspects_cleaned = [a.replace('/', ' ') for a in CORE_ASPECTS]
    core_embeddings = similarity_model.encode(core_aspects_cleaned, convert_to_tensor=True)
    
    # Dedicated ABSA model for context-aware sentiment
    # This model expects inputs in the format: [sentence] [SEP] [aspect]
    absa_pipeline = pipeline(
        "text-classification", 
        model="yangheng/deberta-v3-base-absa-v1.1",
        device=device_idx,
        top_k=None # Get all label probabilities for continuous scoring
    )

    logger.info(f"Starting improved ABSA processing on {len(df)} reviews...")
    
    results = []
    # Threshold for mapping raw terms to core aspects (MiniLM cosine similarity)
    SIMILARITY_THRESHOLD = 0.5 
    
    for text in tqdm(df['text'], desc="Processing reviews"):
        if not isinstance(text, str) or not text.strip():
            results.append([])
            continue
            
        doc = nlp(text)
        review_aspects = []
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
                
            # Step 1: Extract noun chunks that contain nouns or proper nouns
            raw_chunks = [
                chunk.text.strip() 
                for chunk in sent.noun_chunks 
                if any(token.pos_ in ["NOUN", "PROPN"] for token in chunk)
            ]
            
            if not raw_chunks:
                continue
            
            # Step 2 & 3: Map to core aspects and filter noise
            chunk_embeddings = similarity_model.encode(raw_chunks, convert_to_tensor=True)
            cosine_scores = util.cos_sim(chunk_embeddings, core_embeddings)
            
            # Identify which raw chunks map to which core aspects
            mappings = [] # List of (raw_chunk_text, core_aspect_label)
            for i, chunk_text in enumerate(raw_chunks):
                max_score, max_idx = torch.max(cosine_scores[i], dim=0)
                if max_score > SIMILARITY_THRESHOLD:
                    mappings.append((chunk_text, CORE_ASPECTS[max_idx]))
            
            if not mappings:
                continue
                
            # Step 4: Batch score sentiment for mapped aspects within this sentence
            # Use the raw chunk for the ABSA model to ensure it finds the correct context
            absa_inputs = [{"text": sent_text, "text_pair": m[0]} for m in mappings]
            try:
                absa_outputs = absa_pipeline(absa_inputs)
                
                # Robustly ensure absa_outputs is a list of lists of dicts
                if isinstance(absa_outputs, dict):
                    absa_outputs = [absa_outputs]
                if len(absa_outputs) > 0 and not isinstance(absa_outputs[0], list):
                    absa_outputs = [absa_outputs]
                
                for (raw_text, core_label), output in zip(mappings, absa_outputs):
                    # Convert labels to probabilities
                    scores_dict = {res['label'].lower(): res['score'] for res in output}
                    
                    # Continuous score calculation: P(Positive) - P(Negative)
                    # Range: [-1.0, 1.0]
                    pos_prob = scores_dict.get('positive', 0.0)
                    neg_prob = scores_dict.get('negative', 0.0)
                    continuous_score = pos_prob - neg_prob
                    
                    review_aspects.append((core_label, round(continuous_score, 4)))
            except Exception as e:
                logger.warning(f"Error scoring aspects in sentence: {e}")
                continue
        
        results.append(review_aspects)
        
    df['aspect_sentiments'] = results
    logger.info("ABSA processing complete.")
    return df

if __name__ == "__main__":
    try:
        # Load sample data
        logger.info("Loading sample data for verification...")
        data = load_and_clean_data().head(10)
        
        # Run extraction
        processed_data = extract_aspects_and_sentiments(data)
        
        # Print results for verification
        print("\n" + "="*50)
        print("ABSA VERIFICATION RESULTS (Core Aspects & Context-Aware Scores)")
        print("="*50)
        
        for idx, row in processed_data.iterrows():
            print(f"\nReview #{idx+1}:")
            print(f"Text: {row['text'][:150]}...")
            print(f"Aspects & Sentiments: {row['aspect_sentiments']}")
            print("-" * 30)
            
    except Exception as e:
        logger.error(f"An error occurred during standalone execution: {e}")
        import traceback
        traceback.print_exc()
