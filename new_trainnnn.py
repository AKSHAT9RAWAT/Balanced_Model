import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import json
import os

# --------------------------
# 1. Load Dataset
# --------------------------
with open("balanced.json", "r") as f:
    data = json.load(f)

data = {k: v for k, v in data.items() if 'captions' in v and 'label' in v}

captions_list = [album['captions'] for album in data.values()]
labels_list = [album['label'] for album in data.values()]

# --------------------------
# 2. Encode Captions
# --------------------------
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_album_embeddings(captions_list):
    album_embeddings = []
    for album_captions in captions_list:
        embeddings = sentence_model.encode(album_captions, convert_to_tensor=True)
        album_embedding = torch.mean(embeddings, dim=0)
        album_embeddings.append(album_embedding)
    return torch.stack(album_embeddings)

X = get_album_embeddings(captions_list)
y = torch.tensor(labels_list, dtype=torch.float32).view(-1, 1)

# --------------------------
# 3. Train/Test Split
# --------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# 4. Model Definition
# --------------------------
class StoryTriggerModel(nn.Module):
    def __init__(self):
        super(StoryTriggerModel, self).__init__()
        self.fc1 = nn.Linear(384, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# --------------------------
# 5. Training Setup
# --------------------------
model = StoryTriggerModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Create directory before saving any models
os.makedirs("balanced_model", exist_ok=True)

# Early stopping config
best_val_loss = float('inf')
patience = 10
epochs_no_improve = 0
num_epochs = 200  # start safe

# --------------------------
# 6. Training Loop
# --------------------------
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    print(f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {loss.item():.4f}  Val Loss: {val_loss.item():.4f}")

    scheduler.step(val_loss)

    # Early stopping
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        torch.save(model.state_dict(), "balanced_model/balanced_model.pt")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print("⛔ Early stopping triggered.")
        break

# --------------------------
# 7. Evaluation
# --------------------------
model.load_state_dict(torch.load("balanced_model/balanced_model.pt"))
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_val)
    y_pred = y_pred_probs.round()

acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

print(f"\n✅ Accuracy: {acc:.4f}")
print(f"🎯 F1 Score: {f1:.4f}")

# --------------------------
# 8. Save Final Model & Metadata
# --------------------------
torch.save(model.state_dict(), "balanced_model/balanced_trigger_model.pt")
torch.save({
    'input_dim': 384,
    'hidden_dim': 128,
    'model_name': 'all-MiniLM-L6-v2'
}, "balanced_model/balanced_config.pt")

print("✅ Model saved to 'balanced_model/'")

# --------------------------
# 9. Parameter Summary
# --------------------------
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"🔢 Total Trainable Parameters: {total_params}")
