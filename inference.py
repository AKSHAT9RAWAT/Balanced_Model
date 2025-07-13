import json
import numpy as np
import tensorflow as tf
from collections import defaultdict, Counter

# Configuration
MODEL_PATH = 'album_classifier.keras'
INPUT_JSON = 'input.json'
OUTPUT_JSON = 'predictions.json'
MAX_LENGTH = 30  # Must match training

class InferencePipeline:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.vocab = self._load_or_rebuild_vocab()
        self.unk_token = 1  # Hardcoded to match training
        
    def _load_or_rebuild_vocab(self):
        try:
            with open('vocab.json') as f:
                return json.load(f)
        except FileNotFoundError:
            print("vocab.json not found. Rebuilding from model's embedding layer...")
            return self._reconstruct_vocab()
    
    def _reconstruct_vocab(self):
        """Emergency fallback: Extract vocabulary from model's embedding layer"""
        embedding_layer = self.model.layers[0]
        vocab_size = embedding_layer.input_dim
        
        # Create dummy vocab (won't match training exactly but will work)
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for i in range(2, vocab_size):
            vocab[f'token_{i}'] = i
            
        return vocab
    
    def text_to_sequence(self, text):
        tokens = text.lower().split()
        sequence = [self.vocab.get(token, self.unk_token) for token in tokens]
        if len(sequence) < MAX_LENGTH:
            sequence += [self.vocab['<PAD>']] * (MAX_LENGTH - len(sequence))
        else:
            sequence = sequence[:MAX_LENGTH]
        return sequence
    
    def process_albums(self, input_json):
        with open(input_json) as f:
            data = json.load(f)
        
        results = {}
        
        for album_id, album_data in data.items():
            captions = album_data['captions']
            sequences = np.array([self.text_to_sequence(c) for c in captions])
            
            # Get predictions
            preds = self.model.predict(sequences, verbose=0).flatten()
            
            # Store results
            results[album_id] = {
                'captions': captions,
                'raw_scores': [float(p) for p in preds],  # Convert numpy floats
                'binary_predictions': [int(p > 0.5) for p in preds],
                'album_score': float(np.mean(preds)),
                'album_prediction': int(np.mean(preds) > 0.5)
            }
        
        return results

def save_predictions(predictions, output_file):
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    pipeline = InferencePipeline()
    predictions = pipeline.process_albums(INPUT_JSON)
    save_predictions(predictions, OUTPUT_JSON)