# IMDb Sentiment Analysis with Neural Networks

This project performs sentiment analysis on IMDb movie reviews using various deep learning models, including simple neural networks, CNNs, and LSTMs. The main goal is to classify reviews as either **positive** or **negative** based on the text of the reviews.

## Project Overview

The project follows these main steps:

1. **Load IMDb Movie Reviews dataset (50,000 reviews)**: The dataset contains both positive and negative reviews.
2. **Preprocessing**: The reviews are preprocessed by removing special characters, numbers, and unnecessary text. Sentiment labels are converted to binary form (1 for positive and 0 for negative).
3. **GloVe Word Embeddings**: GloVe embeddings are used to convert words into vector representations. An embedding matrix is built for the corpus.
4. **Model Training**: Three different models are trained using deep learning techniques:
   - A simple feedforward **Neural Network (NN)**.
   - A **Convolutional Neural Network (CNN)**.
   - A **Long Short-Term Memory Network (LSTM)**.
5. **Prediction**: The trained models are used to predict the sentiment of real IMDb reviews and evaluate the performance of each model.

## Requirements

- Python 3.x
- Libraries:
  - `tensorflow`
  - `keras`
  - `numpy`
  - `pandas`
  - `sklearn`
  - `matplotlib`
  - `nltk`
  - `GloVe` embeddings

To install the required dependencies, you can run:

```bash
pip install tensorflow keras numpy pandas scikit-learn matplotlib nltk
```

Make sure to also download GloVe embeddings from [GloVe official website](https://nlp.stanford.edu/projects/glove/).

## Dataset

The project uses the IMDb Movie Reviews dataset, which contains 50,000 reviews labeled as positive or negative. The dataset can be downloaded from [here](https://ai.stanford.edu/~amaas/data/sentiment/).

## How to Run

1. **Set Up the Environment**: Install all the required libraries mentioned above.
2. **Download GloVe Embeddings**: Ensure you have the GloVe embeddings loaded correctly.
3. **Run the Notebook**: Execute each step of the Jupyter notebook in sequence to preprocess data, train models, and evaluate results.

## Results

The performance of each model is evaluated based on accuracy and other performance metrics. You can find the detailed comparison of models inside the notebook, along with visualizations of training and testing results.

## Future Improvements

- Add hyperparameter tuning for each model.
- Explore other word embeddings like Word2Vec or FastText.
- Implement more advanced architectures like transformers for better performance.
