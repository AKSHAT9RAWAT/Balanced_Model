# import torch
# import torch.nn as nn
# from sentence_transformers import SentenceTransformer
# import json
# import os

# # --------------------------
# # Model Definition
# # --------------------------
# class StoryTriggerModel(nn.Module):
#     def __init__(self):
#         super(StoryTriggerModel, self).__init__()
#         self.fc1 = nn.Linear(384, 128)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(128, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         return x

# # --------------------------
# # Load Model and Encoder
# # --------------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = StoryTriggerModel()
# model.load_state_dict(torch.load("saved_model/story_trigger_model.pt", map_location=device))
# model.to(device)
# model.eval()

# sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

# # --------------------------
# # Inference Function
# # --------------------------
# def predict_album(captions):
#     with torch.no_grad():
#         embeddings = sentence_model.encode(captions, convert_to_tensor=True)
#         album_embedding = torch.mean(embeddings, dim=0).to(device)
#         output = model(album_embedding.unsqueeze(0))  # [1, 384]
#         prob = output.item()
#         label = int(prob >= 0.75)  # Threshold set to 0.75
#         return label, prob

# # --------------------------
# # Main Inference Loop
# # --------------------------
# if __name__ == "__main__":
#     input_path = "a.json"
#     output_path = "a1_predictions.json"

#     try:
#         with open(input_path, "r") as f:
#             album_data = json.load(f)

#         predictions = {}

#         for album_id, album in album_data.items():
#             captions = album.get("captions", [])
#             if not captions:
#                 print(f"⚠️ Album {album_id} has no captions.")
#                 continue

#             print(f"\n📁 Album ID: {album_id}")
#             print(f"   📜 Captions:")
#             for cap in captions:
#                 print(f"      • {cap}")

#             label, prob = predict_album(captions)
#             predictions[album_id] = {
#                 "predicted_label": label,
#                 "probability": round(prob, 4)
#             }

#             print(f"   ➤ Predicted Label: {label} (Probability: {prob:.4f})")

#         # Save output predictions to file
#         with open(output_path, "w") as f:
#             json.dump(predictions, f, indent=2)

#         print(f"\n✅ All predictions saved to {output_path}")

#     except FileNotFoundError:
#         print(f"❌ File '{input_path}' not found.")
#     except Exception as e:
#         print(f"❌ Error: {str(e)}")


import torch
import torch.nn as nn
import json
from sentence_transformers import SentenceTransformer

# --------------------------
# 1. Load Album Captions (your format)
# --------------------------
with open("a.json", "r", encoding="utf-8") as f:
    album_data = json.load(f)

# --------------------------
# 2. Load Sentence Embedding Model
# --------------------------
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_album_embedding(captions):
    if not captions or not isinstance(captions, list):
        return None
    embeddings = sentence_model.encode(captions, convert_to_tensor=True)
    return torch.mean(embeddings, dim=0)

# --------------------------
# 3. Define Neural Model
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
# 4. Load Trained Weights
# --------------------------
model = StoryTriggerModel()
model.load_state_dict(torch.load("balanced_model/balanced_trigger_model.pt", map_location=torch.device("cpu")))
model.eval()

# --------------------------
# 5. Inference Loop
# --------------------------
results = {}

with torch.no_grad():
    for album_id, album_info in album_data.items():
        captions = album_info.get("captions", [])
        embedding = get_album_embedding(captions)

        if embedding is not None:
            output = model(embedding.unsqueeze(0))  # add batch dimension
            predicted_label = int(output.round().item())

            results[album_id] = {
                "captions": captions,
                "label": predicted_label
            }
        else:
            print(f"⚠️ Skipping album {album_id} due to empty or invalid captions.")

# --------------------------
# 6. Save Prediction Output
# --------------------------
with open("balanced_output.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("✅ Inference complete. Results saved to 'balanced_output.json'")



