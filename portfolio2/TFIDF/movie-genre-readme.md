# Movie Genre Classification with TF-IDF

This project demonstrates movie genre classification using TF-IDF (Term Frequency-Inverse Document Frequency) text vectorization and machine learning models. The system predicts movie genres based on their plot descriptions.

## Overview

The notebook implements a complete text classification pipeline:
- Data loading and preprocessing
- TF-IDF feature extraction
- Multiple model comparison
- Performance evaluation
- Genre prediction from descriptions

## Features

- **Multi-class Classification**: Handles 27 different movie genres
- **TF-IDF Vectorization**: Uses unigrams and bigrams with sublinear term frequency
- **Model Comparison**: Tests multiple classifiers (LinearSVC, Logistic Regression, Naive Bayes, Random Forest)
- **Feature Analysis**: Identifies top correlated n-grams per genre using chi-square statistics
- **Comprehensive Evaluation**: Includes classification reports, confusion matrices, and cross-validation

## Dataset

The dataset (`train_data.txt`) contains movie records with:
- **ID**: Unique identifier
- **Title**: Movie title with year
- **Genre**: One of 27 genres
- **Description**: Plot summary text

Format: `id:::title:::genre:::description`

**Note**: The dataset contains 7,935 movies after cleaning duplicates and invalid entries.

## Requirements

```python
numpy
pandas
matplotlib
scikit-learn
```

## Installation

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Key Results

### Model Performance (3-fold CV)

| Model | Mean Accuracy | Std Dev |
|-------|--------------|---------|
| LinearSVC | 52.77% | 0.63% |
| Logistic Regression | 48.66% | 0.46% |
| Multinomial NB | 43.79% | 0.52% |
| Random Forest | 41.12% | 0.13% |

### Genre Distribution

The dataset is imbalanced with:
- **Documentary**: 481 examples (largest)
- **Drama**: 492 examples
- **Comedy**: 270 examples
- **War**: 5 examples (smallest)

### Top Correlated N-grams

Examples of genre-specific features:
- **Action**: martial, game law, brock lesnar
- **Horror**: horror, undead, vampire
- **Documentary**: documentary, interviews, history
- **Comedy**: comedy, hilarious, sketch comedy

## Usage

### Training the Model

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Create pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        sublinear_tf=True,
        min_df=2,
        stop_words='english',
        ngram_range=(1, 2)
    )),
    ('clf', LinearSVC())
])

# Train
model.fit(X_train, y_train)
```

### Making Predictions

```python
def predict_genre_from_description(description_text: str) -> str:
    pred_id = int(final_model.predict([description_text])[0])
    return str(le.inverse_transform([pred_id])[0])

# Example
description = "A thrilling story about space exploration..."
genre = predict_genre_from_description(description)
print(f"Predicted genre: {genre}")
```

## Model Architecture

1. **TF-IDF Vectorizer**
   - Sublinear term frequency scaling
   - Minimum document frequency: 2
   - English stop words removed
   - Unigrams and bigrams (1-2)

2. **Classifier: LinearSVC**
   - Support Vector Machine with linear kernel
   - Best performing model (52.77% accuracy)
   - Fast training and prediction

## Performance Analysis

### Common Confusions

The model frequently confuses:
- **Drama ↔ Comedy**: 86 misclassifications
- **Drama ↔ Documentary**: 63 misclassifications
- **Short ↔ Documentary**: 62 misclassifications

This is expected as:
- Many genres have overlapping themes
- Some genres are defined by format (Short) rather than content
- Drama is a broad category that overlaps with many others

### Strengths

- High precision for distinctive genres (Western: 88%, Animation: 100%)
- Good recall for Documentary (86%) and Drama (68%)
- Effective at identifying genre-specific vocabulary

### Limitations

- Lower performance on rare genres (War, History, Biography)
- Difficulty distinguishing similar genres (Comedy/Drama)
- Class imbalance affects minority genre predictions

## File Structure

```
.
├── Tf_idf_Movie.ipynb          # Main notebook
├── train_data.txt              # Dataset (not included)
├── README.md                   # This file
└── requirements.txt            # Dependencies
```

## Running the Notebook

1. Ensure you have the dataset file `train_data.txt` in the same directory
2. Install required dependencies: `pip install -r requirements.txt`
3. Open and run `Tf_idf_Movie.ipynb` in Jupyter Notebook or Google Colab
4. Follow the cells sequentially for complete analysis

## Future Improvements

1. **Handle Class Imbalance**: 
   - Use SMOTE or class weights
   - Collect more data for minority genres

2. **Advanced Features**:
   - Include movie title features
   - Add release year information
   - Consider cast/crew metadata

3. **Deep Learning**:
   - Try BERT/RoBERTa embeddings
   - Implement attention mechanisms
   - Multi-label classification (movies can have multiple genres)

4. **Ensemble Methods**:
   - Combine multiple models
   - Stack different vectorization approaches (TF-IDF + Word2Vec)

5. **Hyperparameter Tuning**:
   - Grid search for optimal parameters
   - Cross-validation with more folds
   - Feature selection techniques

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is for educational purposes.

## Acknowledgments

- Dataset source: Movie plot descriptions
- Built with scikit-learn and pandas
- TF-IDF implementation from sklearn

## Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This is a baseline model. Real-world applications would benefit from:
- More sophisticated NLP techniques
- Larger and more balanced datasets
- Domain-specific feature engineering
- Regular model retraining with new data

## Citation

If you use this code in your research, please cite:

```
@misc{movie_genre_classification,
  title={Movie Genre Classification with TF-IDF},
  year={2024},
  howpublished={\url{https://github.com/yourusername/movie-genre-classification}}
}
```
