# Yelp Reviews Sentiment Analysis

A deep learning project for sentiment classification on Yelp Reviews, comparing CNN and LSTM models with Keras/TensorFlow.

## Overview

This project uses the Yelp Reviews dataset from Kaggle to build and evaluate sentiment classification models using deep learning. It demonstrates data preprocessing, text cleaning, tokenization, model training with CNN and LSTM architectures, and evaluation. The project compares the performance of Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks for predicting positive or negative sentiment from review text, including the impact of dropout regularization.

## Dataset

The dataset is sourced from the [Yelp Reviews dataset on Kaggle](https://www.kaggle.com/datasets/omkarsabnis/yelp-reviews-dataset).

The dataset includes:

- `yelp.csv`: Training data with review text and ratings

### Key Features:
- **business_id**: Unique business identifier
- **date**: Date of the review
- **review_id**: Unique review identifier
- **stars**: Rating between 1 and 5
- **text**: Full review text
- **type**: Type of record
- **user_id**: Unique user identifier
- **cool**: Number of cool votes
- **useful**: Number of useful votes
- **funny**: Number of funny votes

Please download `yelp.csv` from the [Yelp Reviews dataset](https://www.kaggle.com/datasets/omkarsabnis/yelp-reviews-dataset) and place it in the project `data/` directory.

## Features

- Data loading and exploration of Yelp Reviews dataset
- Text preprocessing: lowercase, punctuation removal, stopword removal
- Binary sentiment labeling (stars >= 4 as positive)
- Text tokenization and sequence padding
- CNN and LSTM model training with Keras/TensorFlow
- Comparison of baseline and dropout-regularized models
- Visualization of training accuracy and loss curves
- Test set evaluation and accuracy comparison

## Installation

### Prerequisites

- Python 3.10 or higher
- uv (for dependency management)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/whats2000/Yelp-Food-Review-Sentiment-Analysis.git
   cd Yelp-Food-Review-Sentiment-Analysis
   ```

2. Install dependencies using uv:
   ```bash
   uv sync
   ```

3. Activate the virtual environment:
   ```bash
   uv run python --version
   ```

## Usage

### Running the Notebook

1. Launch Jupyter Notebook:
   ```bash
   uv run jupyter notebook
   ```

2. Open `analysis.ipynb` and run the cells sequentially.

### Key Steps in the Notebook

1. **Load Data**: Import and explore the Yelp Reviews dataset
2. **Preprocess Data**: Clean text, remove stopwords, create binary labels, train/test split
3. **Text Vectorization**: Tokenize text, pad sequences for model input
4. **Train CNN Models**: Train baseline and dropout-regularized CNN models
5. **Train LSTM Models**: Train baseline and dropout-regularized LSTM models
6. **Visualize Training**: Plot accuracy and loss curves for all models
7. **Evaluate Models**: Compare test accuracies of CNN and LSTM models with/without dropout

### Direct Execution

You can also run the notebook cells directly using nbconvert:

```bash
uv run jupyter nbconvert --to notebook --execute analysis.ipynb
```

## Dependencies

- `ipykernel`: Jupyter kernel for Python
- `matplotlib`: Plotting and visualization
- `nltk`: Natural language processing toolkit
- `notebook`: Jupyter Notebook
- `numpy`: Numerical computing
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning library for preprocessing and metrics
- `tensorflow`: Deep learning framework for CNN and LSTM models

## Model Performance

The project compares the performance of CNN and LSTM models for sentiment classification on the test set, evaluating both baseline models and versions with dropout regularization (rate=0.7). Accuracy metrics are calculated and compared to assess the effectiveness of each architecture and the impact of dropout on overfitting.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle for hosting the Yelp Reviews dataset
- The open-source community for the deep learning and NLP libraries used
- Omkar Sabnis for providing the dataset


