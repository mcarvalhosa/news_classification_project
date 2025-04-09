# News Classification Project

**Project Title**: News Category Classification (HuffPost Dataset)

## Overview
This project aims to classify news articles into different categories using a **CNN-based deep learning model** and a **classical SVM baseline**. By comparing these two approaches, we demonstrate how text preprocessing, feature engineering (TF-IDF), and hyperparameter tuning can significantly impact classification performance.

## Folder Structure

news_classification_project/
├── data/
│   └── news_category.json       # HuffPost dataset in JSON format
│   └── glove.6B.100d.txt        # .............
├── src/
│   ├── main.py                  # Main Python script
├── .gitignore                   # Git ignore file
├── requirements.txt             # List of Python dependencies
└── README.md                    # This file (project documentation)


## Data
- **Dataset**: [HuffPost News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset)
- **Format**: JSON file containing each article’s category, headline, short description, authors, link, and publication date.
- **Categories**: Up to 42 categories (e.g., POLITICS, WELLNESS, SPORTS, etc.).

> **Note**: If the dataset is large, you may want to sample or filter categories to reduce training time.

## Installation
1. **Clone the Repository**:
   git clone https://github.com/mcarvalhosa/news_classification_project   

2. **Navigate to the Project Folder**:
   cd news_classification_project
   
3. **Install Dependencies**:
   pip install -r requirements.txt
Make sure you have Python 3.7+ installed.

## Usage

1. **Add the Dataset**:  
   - Place your `news_category.json` file into the `data` folder.

2. **Run the Main Script**:
    python src/main.py
   
   This will:
   - Load the dataset.
   - Preprocess the text (tokenization, lowercasing, etc.).
   - Train the SVM baseline with TF-IDF.
   - Train the CNN model.
   - Evaluate and compare results.

3. **(Optional) Use the Notebook**:  
   If you want an interactive environment for exploratory data analysis (EDA), open the provided notebook:
   
   *jupyter notebook notebooks/project.ipynb*
   
   Run cells step-by-step to visualize class distributions, run partial training, etc.

## Methodology

### Data Preprocessing
- **Text Cleaning**: Lowercasing, tokenization, and stopword removal using NLTK.
- **Optional**: Lemmatization or stemming to normalize tokens.
- **Train/Validation/Test Split**: Usually 70/15/15 or 80/10/10.

### Baseline: SVM + TF-IDF
1. **TF-IDF Vectorization**: Convert text into numerical features.
2. **SVM Classifier**: A baseline model using a linear or RBF kernel.
3. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV to find optimal regularization parameters (e.g., `C`, `gamma`).

### Deep Learning: CNN
1. **Embedding**: Trainable embedding layer or pretrained embeddings (e.g., GloVe).
2. **1D Convolution**: Capture local n-gram features.
3. **Pooling**: Reduce sequence length while retaining important features.
4. **Dense Layers**: Final classification with softmax activation.
5. **Hyperparameter Tuning**: Adjust learning rate, batch size, number of filters, kernel size, etc.

## Evaluation & Metrics
- **Accuracy**: Overall percentage of correct predictions.
- **F1-Score**: Especially relevant if certain categories are imbalanced.
- **Confusion Matrix**: Identify which categories are most misclassified.

## Hyperparameter Tuning
- **SVM**: Use `GridSearchCV` (scikit-learn) to try different `C`, kernel types, etc.
- **CNN**: Adjust learning rate, batch size, embedding size, number of Conv1D filters, etc.  
  You can manually adjust these or use advanced tools (e.g., Optuna).

## Results
- After training both models, compare performance metrics.  
- **Typical Findings**: CNN might outperform SVM if enough data and proper tuning are applied, but SVM often trains faster and can be easier to tune for moderate datasets.

## Further Improvements
- **Dimensionality Reduction** (e.g., PCA) on TF-IDF features for SVM.
- **More Sophisticated Embeddings**: Pretrained word embeddings or Transformer-based approaches.
- **Data Augmentation**: If certain categories are underrepresented.

## License
This project uses the **Attribution 4.0 International (CC BY 4.0)** license for the dataset. See the [HuffPost dataset page](https://www.kaggle.com/datasets/rmisra/news-category-dataset) for details. Any code in this repository is available under your chosen license (e.g., MIT, Apache 2.0), which you can specify here.

## Contact
For any questions or suggestions, please feel free to open an issue in the GitHub repository or reach out to [Your Name](mailto:youremail@example.com). 

---

Thank you for checking out the **News Classification Project**! We hope this serves as a solid baseline to explore text categorization techniques using both classical machine learning and deep learning approaches.