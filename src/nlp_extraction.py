import pandas as pd
import spacy
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from sentence_transformers import SentenceTransformer, util
import logging
from tqdm import tqdm
import torch
from datasets import Dataset
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

    # Step 4: Batch score sentiment for mapped aspects using Dataset for efficiency
    logger.info(f"Running ABSA on {len(absa_inputs)} aspect-sentence pairs...")
    
    # To handle sequence pairs with Dataset and Pipeline (streaming), we pre-format the inputs.
    # Most ABSA models accept 'sentence [SEP] aspect' as input.
    # We use the pipeline's tokenizer to get the correct separator.
    sep = absa_pipeline.tokenizer.sep_token
    formatted_inputs = [f"{inp['text']} {sep} {inp['text_pair']}" for inp in absa_inputs]
    
    # Create the Dataset object as requested
    hf_dataset = Dataset.from_dict({"text": formatted_inputs})
    
    absa_scores = []
    # Use KeyDataset to yield individual strings from the dataset to the pipeline.
    # This enables the pipeline's internal batching and streaming.
    for output in tqdm(absa_pipeline(KeyDataset(hf_dataset, "text"), batch_size=32), total=len(hf_dataset), desc="Sentiment Analysis"):
        try:
            # Handle the output structure (top_k=None returns a list of dictionaries for each input)
            if isinstance(output, dict):
                output = [output]
            
            scores_dict = {res['label'].lower(): res['score'] for res in output}
            pos_prob = scores_dict.get('positive', 0.0)
            neg_prob = scores_dict.get('negative', 0.0)
            absa_scores.append(pos_prob - neg_prob)
        except Exception as e:
            logger.warning(f"Error processing ABSA output: {e}")
            absa_scores.append(0.0) # Neutral fallback

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
