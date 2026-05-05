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
    
    1. Extracts raw aspects (noun chunks) using spaCy pipe.
    2. Maps unique raw aspects to core beauty aspects using Sentence-Transformers (MiniLM).
    3. Filters out noise based on a semantic similarity threshold.
    4. Calculates context-aware sentiment scores for each matched aspect using a 
       dedicated DeBERTa-based ABSA model in batches.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing at least a 'text' column.
        
    Returns:
        pd.DataFrame: DataFrame with an additional 'aspect_sentiments' column.
                      Each entry is a list of tuples: [(core_aspect, sentiment_score), ...].
    """
    logger.info("Initializing NLP models (spaCy, Sentence-Transformers, and DeBERTa ABSA)...")
    
    # Load spaCy model for raw aspect (noun chunk) extraction
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError:
        logger.info("Downloading spaCy model 'en_core_web_sm'...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    # Determine execution device
    device_idx = 0 if torch.cuda.is_available() else -1
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Similarity model for aspect categorization
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device=device_str)
    # Use space-separated core aspects for better semantic matching
    core_aspects_cleaned = [a.replace('/', ' ') for a in CORE_ASPECTS]
    core_embeddings = similarity_model.encode(core_aspects_cleaned, convert_to_tensor=True)
    
    # Dedicated ABSA model for context-aware sentiment
    absa_pipeline = pipeline(
        "text-classification", 
        model="yangheng/deberta-v3-base-absa-v1.1",
        device=device_idx,
        top_k=None 
    )

    # Step 1: Extract sentences and noun chunks using spaCy pipe
    logger.info(f"Extracting sentences and noun chunks from {len(df)} reviews using spaCy...")
    raw_data = [] # List of (review_idx, sent_text, chunk_text)
    
    texts = df['text'].fillna("").tolist()
    for i, doc in tqdm(enumerate(nlp.pipe(texts, batch_size=128)), total=len(df), desc="Parsing reviews"):
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            
            for chunk in sent.noun_chunks:
                # Basic check for meaningful noun chunks
                if any(token.pos_ in ["NOUN", "PROPN"] for token in chunk):
                    raw_data.append((i, sent_text, chunk.text.strip()))

    if not raw_data:
        df['aspect_sentiments'] = [[] for _ in range(len(df))]
        return df

    # Step 2 & 3: Map to core aspects and filter noise (batch encode unique chunks)
    logger.info(f"Mapping {len(raw_data)} chunks to core aspects...")
    unique_chunks = list(set(item[2] for item in raw_data))
    chunk_embeddings = similarity_model.encode(unique_chunks, convert_to_tensor=True, show_progress_bar=True)
    cosine_scores = util.cos_sim(chunk_embeddings, core_embeddings)
    
    SIMILARITY_THRESHOLD = 0.35
    chunk_to_aspect = {}
    for i, chunk_text in enumerate(unique_chunks):
        max_score, max_idx = torch.max(cosine_scores[i], dim=0)
        if max_score > SIMILARITY_THRESHOLD:
            chunk_to_aspect[chunk_text] = CORE_ASPECTS[max_idx]
            
    # Filter raw_data and prepare ABSA inputs
    filtered_data = [] # List of (review_idx, core_label, sent_text, chunk_text)
    absa_inputs = []
    for review_idx, sent_text, chunk_text in raw_data:
        if chunk_text in chunk_to_aspect:
            core_label = chunk_to_aspect[chunk_text]
            filtered_data.append((review_idx, core_label, sent_text, chunk_text))
            absa_inputs.append({"text": sent_text, "text_pair": chunk_text})

    if not absa_inputs:
        df['aspect_sentiments'] = [[] for _ in range(len(df))]
        return df

    # Step 4: Batch score sentiment for mapped aspects
    logger.info(f"Running ABSA on {len(absa_inputs)} aspect-sentence pairs...")
    absa_scores = []
    
    # Process ABSA in mini-batches to show progress and avoid memory spikes
    ABSA_BATCH_SIZE = 32 if device_idx >= 0 else 8
    for j in tqdm(range(0, len(absa_inputs), ABSA_BATCH_SIZE), desc="Sentiment Analysis"):
        batch_inputs = absa_inputs[j : j + ABSA_BATCH_SIZE]
        try:
            batch_outputs = absa_pipeline(batch_inputs)
            
            # Normalize outputs to List[List[Dict]]
            if isinstance(batch_outputs, dict): batch_outputs = [[batch_outputs]]
            elif isinstance(batch_outputs, list) and len(batch_outputs) > 0 and isinstance(batch_outputs[0], dict):
                # If top_k=None returns List[Dict] for multiple inputs (happens in some versions/configs)
                # But typically top_k=None returns List[List[Dict]] for multiple inputs.
                # Let's verify by looking at the first element.
                # If it's a list, we're good. If it's a dict, we need to wrap each.
                if isinstance(batch_outputs[0], dict):
                    # Check if it's a single input result or multiple
                    # Actually, if we passed a list, it should be multiple.
                    # This happens if top_k is NOT None, but we set top_k=None.
                    # We'll assume the standard Transformer behavior for top_k=None.
                    pass # Handled below
            
            for output in batch_outputs:
                # Ensure output is a list of results for that input
                if isinstance(output, dict): output = [output]
                
                scores_dict = {res['label'].lower(): res['score'] for res in output}
                pos_prob = scores_dict.get('positive', 0.0)
                neg_prob = scores_dict.get('negative', 0.0)
                absa_scores.append(pos_prob - neg_prob)
        except Exception as e:
            logger.warning(f"Error in ABSA batch: {e}")
            # Fill with neutral scores for failed batch
            absa_scores.extend([0.0] * len(batch_inputs))

    # Step 5: Reassemble results back to reviews
    logger.info("Reassembling results...")
    results = [[] for _ in range(len(df))]
    for (review_idx, core_label, _, _), score in zip(filtered_data, absa_scores):
        results[review_idx].append((core_label, round(score, 4)))
        
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
