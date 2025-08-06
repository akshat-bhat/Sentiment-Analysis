# Sentiment Analysis with Machine Learning

## Project Overview

This project implements a comprehensive sentiment analysis system using Python and machine learning. It compares traditional ML approaches (Logistic Regression) with ensemble methods (Random Forest) and includes modern features like model persistence for instant startup.

**Key Features:**
- **Dual Model Comparison**: Logistic Regression vs Random Forest
- **Smart Model Persistence**: 7+ minute training → 2-3 second startup  
- **Advanced Text Preprocessing**: Lemmatization, stopword removal, social media optimization
- **Comprehensive Evaluation**: ROC curves, confusion matrices, cross-validation
- **Production-Ready**: Confidence scores and easy prediction interface

## Quick Start

### Prerequisites
```
Python >= 3.7
```

### Installation
```
# Clone the repository
git clone 
cd sentiment-analysis

# Create virtual environment
python -m venv sentivenv
sentivenv\Scripts\activate  # Windows
# source sentivenv/bin/activate  # Linux/Mac

# Install dependencies
pip install scikit-learn pandas numpy nltk textblob matplotlib seaborn
```

### Usage
```
# First run (trains and saves models)
python sentiment_analysis.py  # Takes ~7 minutes

# Subsequent runs (loads saved models)
python sentiment_analysis.py  # Takes ~2-3 seconds
```

## Performance Results

| Model | Accuracy | F1-Score | CV Accuracy | Training Time |
|-------|----------|----------|-------------|---------------|
| Logistic Regression | ~85-90% | ~0.85 | ~87% | ~1-2 minutes |
| Random Forest | ~88-92% | ~0.88 | ~89% | ~5-6 minutes |

*Results may vary based on dataset and hardware*

## Technical Implementation

### Architecture
```
Data Input → Text Preprocessing → Feature Extraction → Model Training → Evaluation → Persistence
```

### Text Preprocessing Pipeline
- **Lowercase conversion**
- **URL/mention removal** (social media ready)
- **Punctuation and digit removal**
- **Stopword filtering**
- **Lemmatization** (WordNet)
- **N-gram TF-IDF vectorization** (1,2-grams, 5000 features)

### Models Implemented
1. **Logistic Regression**: Fast, interpretable baseline
2. **Random Forest**: Ensemble method with feature importance

### Evaluation Metrics
- **Accuracy & F1-Score**
- **5-fold Cross-Validation**
- **ROC Curves & AUC**
- **Confusion Matrices**
- **Feature Importance Analysis**

## Project Structure
```
sentiment-analysis/
│
├── sentiment_analysis.ipynb    # Main Jupyter notebook
├── train.csv                   # Training dataset
├── sentiment_models.pkl        # Saved models (auto-generated)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── saved_models/              # Alternative model storage (optional)
```

## Key Innovations

### Smart Model Persistence
```
# Automatic save after training
joblib.dump({'results': results, 'encoder': encoder, 'preprocessor': preprocessor}, 'sentiment_models.pkl')

# Instant load on startup
if os.path.exists('sentiment_models.pkl'):
    data = joblib.load('sentiment_models.pkl')
    # Ready in 2-3 seconds!
```

### Prediction Interface
```
# Easy prediction with confidence
result = predict_sentiment("This product is amazing!")
print(f"Sentiment: {result['prediction_label']} ({result['confidence']})")
```

## Use Cases

- **Customer Feedback Analysis**: Analyze product reviews and support tickets
- **Social Media Monitoring**: Track brand sentiment across platforms
- **Content Moderation**: Identify negative or toxic content
- **Market Research**: Understand public opinion on topics

## Development Workflow

### To Retrain Models
1. Delete `sentiment_models.pkl`
2. Run notebook again
3. Models will retrain and save automatically

### To Add New Features
1. Modify preprocessing in `TextPreprocessor` class
2. Add new models to `create_models()` function
3. Update evaluation metrics as needed

## Performance Optimization

**Hardware Requirements:**
- **Minimum**: 4GB RAM, 2-core CPU
- **Recommended**: 8GB+ RAM, 4+ core CPU
- **Training Time**: ~7 minutes (tested on mid-range laptop)
- **Inference Time**:  stemming
- **TF-IDF Explained**: Mathematical intuition behind vectorization

**Next Steps:**
- **Transformer Models**: BERT, RoBERTa for advanced NLP
- **Deep Learning**: PyTorch/TensorFlow for neural networks
- **MLOps**: Model deployment and monitoring

## Author

Created as part of machine learning self-study journey. Focus on creating explainable, reverse-engineerable code for deep understanding.


**⭐ If this project helped you learn sentiment analysis, please give it a star!**






