import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, precision_recall_curve

# 1. Data Preparation
def load_and_prepare_data(json_path):
    with open(json_path) as f:
        data = json.load(f)
    
    # Prepare dataset
    X, y = [], []
    for album in data.values():
        X.extend(album['captions'])
        y.extend([album['label']] * len(album['captions']))
    
    # Check class balance
    label_counts = Counter(y)
    print(f"Class distribution: {label_counts}")
    plt.bar(['Not Meaningful (0)', 'Meaningful (1)'], 
            [label_counts[0], label_counts[1]],
            color=['red', 'green'])
    plt.title('Class Distribution')
    plt.savefig('class_distribution.png')
    plt.close()
    
    return np.array(X), np.array(y)

# 2. Text Processing
def build_vocabulary(texts, max_words=8000):
    word_counts = Counter()
    for text in texts:
        tokens = text.lower().split()
        word_counts.update(tokens)
    
    vocab = {word: i+2 for i, (word, count) in enumerate(word_counts.most_common(max_words-2))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

def text_to_sequence(text, vocab, max_length=30):
    tokens = text.lower().split()
    sequence = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    if len(sequence) < max_length:
        sequence += [vocab['<PAD>']] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]
    return sequence

# 3. Model Architecture
def build_model(vocab_size, max_length=30):
    model = tf.keras.Sequential([
        layers.Embedding(vocab_size, 128, input_length=max_length, mask_zero=True),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc', curve='PR'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model

# 4. Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, (y_pred > 0.5).astype(int)))
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('precision_recall_curve.png')
    plt.close()

# Main Training Function
def train():
    # Load data
    X, y = load_and_prepare_data('caption_groundtruth.json')
    X, y = shuffle(X, y, random_state=42)
    
    # Build vocabulary
    vocab = build_vocabulary(X)
    vocab_size = len(vocab)
    max_length = 30
    
    # Convert texts to sequences
    X_seq = np.array([text_to_sequence(text, vocab, max_length) for text in X])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build model
    model = build_model(vocab_size, max_length)
    
    # Class weights
    total = len(y_train)
    weight_for_0 = (1 / np.sum(y_train == 0)) * (total / 2.0)
    weight_for_1 = (1 / np.sum(y_train == 1)) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    # Train
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=5,
                mode='max',
                restore_best_weights=True
            )
        ]
    )
    
    # Save and evaluate
    model.save('album_classifier.keras')
    evaluate_model(model, X_test, y_test)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print("\nTraining complete. Model saved as 'album_classifier.keras'")

if __name__ == "__main__":
    train()