Emotion Classification with Deep Learning, Attention & DistilBERT
A comparative NLP project that classifies English text into six basic emotions using recurrent neural networks, an attention-based BiLSTM, and a fine-tuned transformer. The project uses GoEmotions, evaluates models on common validation and test partitions, visualizes learned attention weights, and provides an interactive Gradio interface.
Emotions
anger · disgust · fear · joy · sadness · surprise
Project highlights
- Map GoEmotions' 28 original labels (27 emotions plus neutral) to six Ekman emotion groups using the official mapping; retain examples mapping unambiguously to exactly one target emotion.
- Clean and tokenize text; train word-based models with 100-dimensional pretrained GloVe embeddings.
- Compare LSTM, GRU, BiLSTM + additive attention, and fine-tuned DistilBERT (distilbert-base-uncased).
- Address class imbalance with class weights for the recurrent models.
- Measure accuracy, macro-precision, macro-recall, macro-F1, and per-class metrics; generate confusion matrices.
- Visualize BiLSTM attention scores over input tokens (as a diagnostic, not a causal explanation).
- Select the deployment model based on validation macro-F1 and export artifacts for a standalone Gradio app.
Dataset
Source: Google GoEmotions — simplified split

The original dataset contains 28 labels, including neutral, and may assign multiple labels to an example. This project retains examples whose non-neutral labels map to one of the six target groups; neutral-only and cross-group ambiguous examples are excluded.
Split	Final samples
Train	28,102
Validation	3,527
Test	3,539


The notebook initially reports 28,104 train examples after label mapping, then removes two during cleaning, leaving 28,102 for training.

Note: Classes are imbalanced, with joy dominant and fear/disgust relatively rare. This is why macro-F1 and per-class recall are important alongside accuracy.
Method
GoEmotions (simplified)
    -> Ekman label mapping + single-target filtering
    -> Text preprocessing
    -> +------------------------------+
       | Word tokenizer + GloVe 100d |
       |   LSTM                       |
       |   GRU                        |
       |   BiLSTM + Additive Attention|
       +------------------------------+
    -> DistilBERT tokenizer + fine-tuning (separate transformer branch)
    -> Validation/test evaluation + confusion matrices
    -> Model selection by validation macro-F1
    -> Gradio inference + optional attention visualization
The word-based models use a maximum sequence length of 24 tokens, a vocabulary capped at 20,000, and trainable GloVe-initialized embeddings. The transformer uses its own tokenizer and fine-tuning pipeline.
Results
Held-out test split, as recorded in the supplied notebook:
Model	Accuracy	Macro-F1
LSTM	68.66%	0.5913
GRU	69.68%	0.6137
BiLSTM + Attention	68.49%	0.6022
DistilBERT	81.27%	0.7031


Validation macro-F1: LSTM 0.5691; GRU 0.5921; BiLSTM + Attention 0.5812; DistilBERT 0.7255. DistilBERT was selected based on validation performance rather than the test split.

The attention-based model offers token-level visual inspection but does not outperform GRU in this experiment. The results should not be interpreted as a general comparison of all possible model configurations.
Running the notebook
1. Open emotion-classification-nlp.ipynb in Kaggle with internet access and a GPU-enabled runtime.
2. Run the cells in order; the notebook installs required libraries, downloads GoEmotions and GloVe, trains four models, evaluates them, exports artifacts, and launches Gradio.
3. For deployments, retain the exported model weights/tokenizers and metadata; the notebook alone is not a replacement for those trained artifacts.
Libraries: Python, TensorFlow/Keras, PyTorch, Hugging Face transformers and datasets, Gensim, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, Gradio.

Kaggle memory note: The notebook limits TensorFlow GPU memory to 3 GiB before any TensorFlow model operations to leave memory for PyTorch/DistilBERT. This setting is environment-specific and may require a fresh kernel and adjustments on different hardware. Its documentation states that it was syntax-checked, not independently re-run end-to-end after the memory-fix changes, although outputs from an earlier execution are included.
Standalone Gradio deployment
During execution, the notebook generates app.py, model files, the selected DistilBERT checkpoint, tokenizer and label mappings, reports, and a ZIP archive.

Place the exported app.py alongside its expected artifacts and install compatible dependencies. Then run:
python app.py
The Gradio interface accepts English text and can display the predicted emotion, probabilities for all six classes, and an attention heatmap when the BiLSTM + Attention model is selected. Do not expect standalone inference to work from the notebook file alone: the saved weights and preprocessing artifacts must be available.

Typical generated artifacts include:
app.py
keras_tokenizer.pkl
label_maps.json
lstm_model.keras
gru_model.keras
bilstm_attention_model.keras
bilstm_attention_weights_model.keras
distilbert_emotion_final/
emotion_results/
The notebook generates a emotion_models_export.zip of approximately 329 MB; consider Git LFS or external model hosting rather than uploading the archive directly to a conventional Git repository.
Evaluation caveats and limitations
- Text overlap across splits: The notebook reports 53 overlapping cleaned texts between train/validation, 52 between train/test, and 11 between validation/test. This means the reported test scores are not certified leakage-free. A deduplicated resplit and rerun are needed before making that claim.
- Class imbalance: Joy is prevalent; fear and disgust have relatively few samples. Report macro-F1 and minority-class recall rather than accuracy alone.
- Single-target conversion: Removing neutral and ambiguous examples changes the original GoEmotions task, so these scores are not directly comparable with original 28-label multilabel benchmarks.
- Attention interpretation: Attention highlights token weights in the BiLSTM model but does not prove which words causally determined a prediction.
- Reproducibility: The memory-adjusted notebook needs a full rerun in the target runtime before results can be considered independently verified for that version.
Possible next steps
- Remove normalized-text duplicates across train, validation and test, then retrain and re-evaluate.
- Report uncertainty (multiple seeds or confidence intervals), and examine minority-emotion errors.
- Add pinned dependency versions, automated smoke tests, and model hosting for reproducible deployment.
Dataset attribution
This project uses GoEmotions, released by Google Research. Review its dataset terms and attribution requirements before republishing data. This repository is an educational, applied NLP model comparison, not a claim of a newly invented model architecture.
