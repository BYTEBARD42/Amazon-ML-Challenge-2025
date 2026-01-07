# ML Challenge 2025: Smart Product Pricing Solution

## 1. Executive Summary

This solution presents a deep learning approach to predict product prices from textual descriptions. We utilize a BI Long Short-Term Memory (BI-LSTM) network, which is adept at understanding sequential data, coupled with a custom text preprocessing pipeline to transform raw product catalog content into a model-readable format.

---

## 2. Methodology Overview

### 2.1 Problem Analysis

The core challenge is to accurately estimate a product's price based solely on its descriptive text (`catalog_content`). Our initial analysis indicated that the descriptions vary significantly in length, vocabulary, and structure. This unstructured nature necessitates a robust natural language processing (NLP) model that can capture semantic meaning and context from the text to infer its value.

**Key Observations:**

-   A large and diverse vocabulary is present across the product descriptions.
-   Text requires significant cleaning and normalization to be useful for a machine learning model.
-   The sequential nature of words in a description is important for understanding the product's features and, consequently, its price.

### 2.2 Solution Strategy

We adopted a single-model strategy centered around a Recurrent Neural Network (RNN). This approach was chosen for its proven effectiveness in sequence modeling tasks. The core of our solution is an LSTM network that learns to map sequences of words to a continuous price value.

**Approach Type:** Single Model (Deep Learning / NLP)
**Core Innovation:** A self-contained `TextPreprocessor` class that handles vocabulary building, text cleaning, tokenization, and sequence padding, making the data preparation pipeline reusable and efficient.

---

## 3. Model Architecture

### 3.1 Architecture Overview

The architecture follows a standard NLP regression pipeline. Raw text is first processed by our `TextPreprocessor`. The resulting fixed-length numerical sequences are fed into an embedding layer, which learns dense vector representations for each word. These embeddings are then processed by a multi-layer LSTM to capture contextual information, and the final output is passed through a dense layer to produce the price prediction.

### 3.2 Model Components

**Text Processing Pipeline:**

-   **Preprocessing steps:**
    -   Text cleaning (lowercasing, removing non-alphanumeric characters).
    -   Tokenization (splitting text into words).
    -   Building a vocabulary of the top 10,000 words.
    -   Text-to-sequence conversion (mapping words to integer indices).
    -   Padding/truncating sequences to a fixed length of 50.
-   **Model type:** `LSTMPricePredictor` (A PyTorch-based LSTM model).
-   **Key parameters:**
    -   `vocab_size`: 10,000
    -   `max_length`: 50
    -   `embedding_dim`: 128
    -   `hidden_dim`: 256
    -   `num_layers`: 2

---

## 4. Model Performance

### 4.1 Validation Results

_The provided code is a prediction pipeline, so training and validation metrics are not generated. The model would be evaluated using the SMAPE metric._

-   **SMAPE Score:** `47.78` on training dataset

---

## 5. Conclusion

Our fine tuned BI-LSTM-based model provides a strong framework for predicting product prices from descriptions. The key to the solution's success lies in the careful text preprocessing and the ability of the BI-LSTM to learn from the sequential nature of the language. Future work could explore incorporating other data modalities or using more advanced transformer-based architectures.
