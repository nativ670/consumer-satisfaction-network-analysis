# AI INSTRUCTIONS & PROJECT PLAN: Consumer Satisfaction Network Analysis

## 1. Project Overview & Research Question
**Theoretical Goal:** To understand how different experience attributes (e.g., price, service, quality) are represented and weighted in consumers' mental models. Specifically, do consumers judge experiences based on isolated, independent features, or do they rely on complex behavioral schemas driven by significant interactions and cognitive trade-offs?
**Operational Goal:** To determine if incorporating measured network interactions between experience features improves the accuracy of predicting overall customer satisfaction, compared to a traditional regression model that treats features independently.

## 2. Psychological & Behavioral Context
This project seeks to prove that human evaluation of experiences is non-linear. The code should help us identify behavioral "trade-offs". For example, finding that a highly positive feature (like "great location") dramatically compensates for a negative feature (like "bad service"), thereby changing the weights consumers assign to these attributes dynamically based on context.

## 3. Data Source & Handling
* **Dataset:** Amazon Reviews 2023 Dataset from Hugging Face (`McAuley-Lab/Amazon-Reviews-2023`).
* **Domain/Subset:** `raw_review_All_Beauty`.
* **Strict Data Rule:** DO NOT download or save the dataset to disk (no local CSVs). Use the `datasets` library to load the data directly into memory or stream it from the Hugging Face API dynamically.

## 4. Architecture & Pipeline (`src/` directory)
The project is divided into specific modules. When asked to work on a file, focus only on its designated role:

* **`src/data_preprocessing.py`:** Contains a reusable data loader function that fetches the data from Hugging Face in-memory. It should filter for essential columns (`rating`, `title`, `text`, `user_id`, `parent_asin`) and clean missing values.
* **`src/nlp_extraction.py`:** Uses Natural Language Processing to extract key experience attributes from the `text` column and assigns a sentiment score (positive/negative) to each extracted aspect.
* **`src/network_builder.py`:** Represents the extracted features as a network of nodes and edges. It must calculate behavioral network metrics such as Centrality and Tie Strength to quantify the interactions between features.
* **`src/modeling.py`:** Builds two distinct statistical models to predict the overall `rating` (satisfaction):
    1.  *Baseline Additive Model:* A standard regression model assuming independent feature effects.
    2.  *Network Model:* A model incorporating the network metrics (interactions/tie strength) calculated in the previous step.
* **`src/evaluation.py`:** Compares the two models using K-fold cross-validation. It must calculate and output predictive error (`RMSE`), explained variance (`Adjusted R^2`), and model complexity (`BIC`). It should also generate visualizations highlighting the trade-offs between features.

## 5. Coding Guidelines
* **Libraries:** Primarily use `pandas`, `scikit-learn`, `networkx`, `statsmodels`, and `datasets`. Use standard NLP libraries (`spacy` or `transformers`) as appropriate.
* **Documentation:** Every function must have clear Python docstrings explaining its parameters and return values.
* **Modularity:** Scripts should be importable. Use `if __name__ == "__main__":` blocks for CLI execution or testing, allowing functions to be imported seamlessly across the pipeline.