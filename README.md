# Emotion Classification Using Deep Learning and Transformers

## Project Overview

This project focuses on classifying emotions expressed in text using Natural Language Processing (NLP) and Deep Learning.

The goal is to compare different neural network architectures and identify which model performs best at recognizing emotions.

Four models were implemented and evaluated:
- LSTM
- GRU
- BiLSTM with Attention
- DistilBERT

An interactive **Gradio application** was also developed to allow users to enter text and predict its emotion.

## Dataset

The project uses the **GoEmotions dataset**, which contains text samples labeled with human emotions.

The following six emotions are considered:

- Joy
- Sadness
- Anger
- Fear
- Surprise
- Disgust

The text data was cleaned, preprocessed, and prepared for model training and evaluation.

## Models

**1. LSTM**

A recurrent neural network designed to learn relationships between words in a sequence.

**2. GRU**

A simpler recurrent architecture that uses fewer parameters than LSTM while maintaining competitive performance.

**3. BiLSTM with Attention**

Processes text in both directions and uses an attention mechanism to highlight important words when predicting emotions.

**4. DistilBERT**

A pretrained Transformer model fine-tuned for emotion classification.

## Results

The models were evaluated using Accuracy and Macro-F1 Score.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| LSTM | 68.66% | 0.5913 |
| GRU | 69.68% | 0.6137 |
| BiLSTM + Attention | 68.49% | 0.6022 |
| **DistilBERT** | **81.27%** | **0.7031** |

**DistilBERT achieved the best overall performance**, showing the effectiveness of pretrained Transformer models for emotion classification.

## Gradio Application

An interactive interface was developed using Gradio.

Users can:
- Enter a sentence or short text.
- Predict the emotion expressed in the text.
- View the model's confidence score.

**Example**

Input: `I am so happy with my results!`

Predicted Emotion: `Joy`

## How to Run

1. Download or clone this repository.
2. Open `emotion-classification-nlp.ipynb` in Kaggle or Jupyter Notebook.
3. Install the required Python libraries.
4. Run the notebook cells in order.
5. Launch the Gradio interface to test emotion predictions.

**Main Technologies**

Python · TensorFlow/Keras · PyTorch · Hugging Face Transformers · Scikit-learn · Gradio

## Key Findings

- DistilBERT achieved the highest Accuracy and Macro-F1.
- GRU performed slightly better than LSTM and BiLSTM with Attention in Macro-F1.
- The Attention mechanism provides useful visualizations of token importance but does not necessarily improve classification accuracy.
- Class imbalance and overlapping text samples across dataset splits are limitations to consider when interpreting the results.

## Conclusion

This project demonstrates how traditional recurrent neural networks and modern Transformer architectures can be applied to text emotion classification.

It also provides hands-on experience in NLP preprocessing, model training, performance evaluation, attention visualization, and interactive deployment.
