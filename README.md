# Balanced Model - Album Caption Classifier

This project implements a machine learning model to classify album captions as meaningful or not meaningful using TensorFlow and natural language processing techniques.

## Project Overview

The model uses a bidirectional LSTM neural network to classify album captions into two categories:

- **Meaningful (1)**: Captions that provide meaningful information about the album
- **Not Meaningful (0)**: Captions that don't provide useful information

## Features

- **Text Processing**: Custom vocabulary building and text-to-sequence conversion
- **Neural Network**: Bidirectional LSTM with embedding layer
- **Class Balancing**: Handles imbalanced datasets with class weights
- **Evaluation**: Comprehensive metrics including precision, recall, and AUC
- **Visualization**: Training history plots and precision-recall curves

## Files Description

- `train.py`: Main training script for the album classifier
- `new_trainnnn.py`: Updated training script with improvements
- `inference.py`: Script for making predictions on new data
- `new_inferenceeee.py`: Enhanced inference script
- `evaluate.py`: Model evaluation and metrics calculation
- `requirements.txt`: Python dependencies
- `caption_groundtruth.json`: Original dataset
- `new_caption_groundtruth.json`: Updated dataset
- `balanced.json`: Balanced dataset for training

## Installation

1. Clone the repository:

```bash
git clone https://github.com/AKSHAT9RAWAT/Balanced_Model.git
cd Balanced_Model
```

2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate the virtual environment:

```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python train.py
```

### Making Predictions

```bash
python inference.py
```

### Evaluating the Model

```bash
python evaluate.py
```

## Model Architecture

The model consists of:

- Embedding layer (128 dimensions)
- Bidirectional LSTM (64 units)
- Global max pooling
- Dense layer with ReLU activation
- Dropout layer (0.5)
- Output layer with sigmoid activation

## Performance Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- AUC (Area Under Curve)
- Precision-Recall curves

## Dataset

The project uses a custom dataset of album captions with binary labels indicating whether each caption is meaningful or not.

## License

This project is part of the Balanced Model repository.
