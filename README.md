# News Classification Project

**Project Title**: News Category Classification (HuffPost Dataset)

## Overview
This project aims to classify news articles into different categories using a **CNN-based deep learning model** and a **classical SVM baseline**. I also explored **LSTM** and **GRU** architectures to compare performance across traditional and sequential models. The project emphasizes the importance of text preprocessing, feature engineering (TF-IDF), dimensionality reduction (TruncatedSVD), and hyperparameter tuning in building effective classification models.

## Folder Structure

news_classification_project/
├── data/
│   ├── news_category_dataset.json     # HuffPost dataset in JSON format  
│   ├── glove.6B.100d.txt              # Pretrained GloVe embeddings (100D)
├── src/
│   ├── main.ipynb                     # Main Python notebook script
│   ├── main.html                      # HTML notebook version
│   ├── cleaned_dataset.csv            # Full preprocessed dataset
│   ├── cleaned_sample_dataset.csv     # Sample (50k) dataset for modeling
├── .gitignore                         # Git ignore file
├── requirements.txt                   # List of Python dependencies
└── README.md                          # This file (project documentation)


## Data
- **Dataset**: [HuffPost News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- **Format**: JSON file containing each article’s category, headline, short description, authors, link, and publication date.
- **Categories**: Initially 42 categories. After merging semantically similar categories (e.g., “ARTS” + “CULTURE & ARTS”), I worked with 28 final classes.


## Installation
1. **Clone the Repository**:
   git clone https://github.com/mcarvalhosa/news_classification_project   

2. **Navigate to the Project Folder**:
   cd news_classification_project
   
3. **Install Dependencies**:
   pip install -r requirements.txt
   
   Make sure you have Python 3.7+ installed.
   
4. **Manually download the spaCy model and GloVe embeddings**:
   - python -m spacy download en_core_web_sm
   - Download GloVe 6B 100d and place it in /data.


## Usage

1. **Add the Dataset**:  
   - Place your `news_category.json` file into the `data` folder.
   - Place your `GloVe embeddings` (glove.6B.100d.txt) into the same `data` folder.

2. **Run the Main Script**:
    python src/main.py
   
   This will:
   - Load the dataset.
   - Visualize category distributions.
   - Preprocess the text (tokenization, lowercasing, etc.).
   - Train and evaluate models (SVM, CNN, LSTM, GRU)
   - Perform hyperparameter tuning using Keras Tuner
   - Compare model performance and display confusion matrices
   
   Run cells step-by-step to visualize class distributions, run partial training, etc.


## Methodology

### 1. Data Preprocessing

- **Text Cleaning**: Lowercased text, removed punctuation, and applied lemmatization using spaCy.
- **Stopword Removal**: Used NLTK and spaCy’s default stopword lists to reduce noise.
- **Category Merging**: Combined semantically similar labels (e.g., "STYLE" + "STYLE & BEAUTY") to reduce class fragmentation and improve learning. I then removed the rare category "LATINO VOICES", resulting in 28 final classes.
- **Empty Text Handling**: Removed 5 entries with empty strings in the `text` column (despite not being null).
- **Train/Validation/Test Split**: Used a 70/15/15 stratified split to maintain category proportions across all subsets.

---

### 2. Models Tested

#### 2.1 Baseline: SVM (TF-IDF + TruncatedSVD)

- **TF-IDF Vectorization**:
  - Used unigrams and bigrams (`ngram_range=(1, 2)`)
  - Removed rare words (`min_df=5`) and overly common ones (`max_df=0.8`)
  - Limited vocabulary to 10,000 features
- **Dimensionality Reduction**:
  - Used TruncatedSVD with 300 components (instead of PCA) to preserve structure in sparse matrices
- **Linear SVM**:
  - Trained with `class_weight="balanced"` to address imbalance
  - Tuned `C` using `GridSearchCV` with 3-fold cross-validation

SVM served as a strong classical baseline, but its reliance on sparse TF-IDF vectors limits its ability to capture deep semantic meaning in text. This motivated the use of deep learning models for richer representation.


#### 2.2 Deep Learning: CNN, LSTM, GRU

- **Embeddings**:
  - Loaded GloVe 100D pretrained vectors
  - Created an embedding matrix and set `trainable=True` to allow fine-tuning
- **CNN Architecture**:
  - Embedding → Conv1D(256 filters, kernel size=3) → GlobalMaxPool1D → Dense(32) → Dropout(0.3) → Output(28)
  - Trained with early stopping and class weights
- **LSTM & GRU**:
  - Replaced Conv1D with LSTM(64) and GRU(64) layers, followed by similar dropout and dense layers
  - Used same tokenized and padded input sequences
  - Sequences were padded/truncated to a maximum length of 100 tokens based on EDA analysis (95th percentile at 55 words), balancing coverage and avoiding over-padding.
  - Trained with early stopping and class weights

#### 2.3 CNN Hyperparameter Tuning (Keras Tuner)

- Tuned key hyperparameters:
  - `embedding_dim`: [50, 100, 200]
  - `filters`: [64, 128, 256]
  - `kernel_size`: [3, 4, 5]
  - `dropout`: 0.3–0.6
  - `dense_units`: [32, 64, 128]
- Used `RandomSearch` with validation accuracy as the optimization objective
- Best model used:
  - `embedding_dim=200`, `filters=256`, `kernel_size=3`, `dropout=0.5`, `dense_units=32`

---

### 3. Evaluation & Metrics

- **Accuracy**: Overall test accuracy
- **Weighted F1 Score**: Adjusted for class imbalance
- **Confusion Matrix**: For visualizing prediction confusion
- **Classification Report**: Formatted into a pandas DataFrame for sorting
- **F1 Score Bar Chart**: Displayed per category to identify best/worst performing labels

---

### 4. Results Summary

The classical SVM model (TF-IDF + SVD) served as a baseline, and all deep learning models were evaluated relative to it. This allowed for a fair performance benchmark using consistent datasets and preprocessing steps.

| Model               | Accuracy | Weighted F1 Score |
|---------------------|----------|-------------------|
| SVM (TF-IDF + SVD)  | 0.538    | 0.546             |
| CNN (Tuned)         | 0.610    | 0.592             |
| LSTM                | 0.633    | 0.621             |
| GRU                 | 0.627    | 0.601             |


- **LSTM** achieved the best overall performance
- Sequential models (LSTM, GRU) slightly outperformed CNN in both accuracy and F1 score.
- Categories like **WELLNESS**, **STYLE & BEAUTY**, and **TRAVEL** had high F1 scores.
- Harder-to-predict categories included **MONEY**, **SCIENCE**, and **IMPACT**.

---

### 5. Further Improvements

- **Try Transformer Models**:
  - Incorporate BERT or DistilBERT for better contextual understanding
- **Upsample or Augment Low-Frequency Classes**:
  - Use paraphrasing, synonym replacement, or back-translation
- **Interpretability Tools**:
  - Apply SHAP or LIME to visualize which words influenced predictions



## License
This project uses the **Attribution 4.0 International (CC BY 4.0)** license for the dataset. See the [HuffPost dataset page](https://www.kaggle.com/datasets/rmisra/news-category-dataset) for details. Any code in this repository is available under your chosen license (e.g., MIT, Apache 2.0), which you can specify here.

## Acknowledgements
- **HuffPost News Dataset** for the source data.
- **GloVe** for pretrained word embeddings.
- The community behind **spaCy** and **Keras** for their excellent tools and support.

## Contact
Martim Carvalhosa – MSc in Analytics and Management @ London Business School
For any questions or suggestions, please feel free to open an issue in the GitHub repository or reach out to [Martim Carvalhosa](mailto:mcarvalhosa.mam2025@london.edu). 

---

Thank you for checking out the **News Classification Project**! I hope this serves as a solid baseline to explore text categorization techniques using both classical machine learning and deep learning approaches.
