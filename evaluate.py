import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Define the model class
class StoryTriggerModel(nn.Module):
    def __init__(self):
        super(StoryTriggerModel, self).__init__()
        self.fc1 = nn.Linear(384, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = StoryTriggerModel()
model.load_state_dict(torch.load("new_saved_model/new_story_trigger_model.pt", map_location=device))
model.to(device)
model.eval()

# Sentence encoder
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# Prediction function
def predict_album(captions):
    with torch.no_grad():
        embeddings = sentence_model.encode(captions, convert_to_tensor=True)
        album_embedding = torch.mean(embeddings, dim=0).to(device)
        output = model(album_embedding.unsqueeze(0))
        prob = output.item()
        label = int(prob >= 0.5)
        return label, prob

# Load ground truth test data
with open("new_caption_groundtruth.json", "r") as f:
    data = json.load(f)

true_labels = []
predicted_labels = []

# Loop over all albums
for album_id, album in data.items():
    if "label" not in album:
        print(f"⚠️ Skipping album {album_id} (no label)")
        continue

    captions = album.get("captions", [])
    if not captions:
        print(f"⚠️ Skipping album {album_id} (no captions)")
        continue

    true_label = album["label"]
    pred_label, prob = predict_album(captions)

    true_labels.append(true_label)
    predicted_labels.append(pred_label)

    print(f"📁 {album_id} → True: {true_label}, Pred: {pred_label}, Prob: {prob:.4f}")

# Show metrics
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print("\n📊 Evaluation Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Plot confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
