## NLP with Disaster Tweets

This repository contains a solution for the [Kaggle NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) competition, where the objective is to build a machine learning model that classifies whether a tweet is related to a real disaster or not.

### 📌 Project Overview

Twitter is a crucial communication channel during emergencies. However, not all tweets mentioning disasters are real; some may be metaphorical. This project aims to develop a Natural Language Processing (NLP) model that can accurately classify tweets as disaster-related (1) or not (0).

### 📂 Repository Structure

```
📁 NLP-with-disaster-tweets
│── 📜 Exploration.ipynb        # Exploratory data analysis (EDA)
│── 📜 Model_Training.ipynb     # Model training and evaluation
│── 📜 dataset.pkl              # Preprocessed dataset (pickled format)
│── 📜 test_dataset.pkl         # Preprocessed test dataset
│── 📜 README.md                # Project documentation
```

### 📊 Dataset

The dataset consists of 10,000 labeled tweets and includes the following columns:

- `id` - Unique identifier for each tweet
- `text` - The tweet content
- `location` - The location from which the tweet was sent (may be missing)
- `keyword` - A keyword extracted from the tweet (may be missing)
- `target` - Label indicating whether the tweet is related to a real disaster (1) or not (0)

### 🚀 Model Training Approach

1. **Data Preprocessing**:
   - Text cleaning (removing special characters, stopwords, etc.)
   - Tokenization and vectorization (TF-IDF, word embeddings)
   - Handling missing data in `keyword` and `location`

2. **Feature Engineering**:
   - Using NLP techniques such as word embeddings and sentiment analysis

3. **Model Selection**:
   - Tried multiple classifiers such as Logistic Regression, Random Forest, LSTM, and Transformers (e.g., RoBERTa, DistilBERT)
   - Evaluated using **F1-score**, Precision, and Recall

### 📈 Evaluation Metric

The competition is evaluated using **F1-score**, calculated as:

$$
F1 = \frac{2 \times \text{precision} \times \text{recall}}{\text{precision} + \text{recall}}
$$

Where:
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- TP: True Positives, FP: False Positives, FN: False Negatives

### 🔧 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Jerryson520/NLP-with-disaster-tweets.git
   cd NLP-with-disaster-tweets
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook and run the `Model_Training.ipynb` file.
