# # # import json
# # # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# # # import torch

# # # # ----------------------------
# # # # Load FLAN-T5 Small
# # # # ----------------------------
# # # model_name = "google/flan-t5-small"
# # # tokenizer = AutoTokenizer.from_pretrained(model_name)
# # # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # # device = "cuda" if torch.cuda.is_available() else "cpu"
# # # model = model.to(device)

# # # # ----------------------------
# # # # Load Input Album Data
# # # # ----------------------------
# # # input_json_path = "glove_good.json"
# # # output_json_path = "TEST_VAL/insta_captions.json"

# # # with open(input_json_path, "r") as f:
# # #     data = json.load(f)

# # # results = {}

# # # # ----------------------------
# # # # Generate Instagram Captions
# # # # ----------------------------
# # # for album_id, album in data.items():
# # #     if album.get("label", 0) != 1:
# # #         continue

# # #     captions = album.get("captions", [])
# # #     if not captions:
# # #         continue

# # #     context = ", ".join(captions[:10])  # Limit to top 10 captions to avoid overflow
# # #     prompt = f"Write a short, emotional Instagram caption for this moment: {context}"

# # #     inputs = tokenizer(prompt, return_tensors="pt").to(device)
# # #     outputs = model.generate(**inputs, max_new_tokens=30)

# # #     insta_caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # #     results[album_id] = {
# # #         "instagram_caption": insta_caption,
# # #         "original_captions": captions
# # #     }

# # #     print(f"Album ID: {album_id}")
# # #     print(f"Instagram Caption: {insta_caption}")
# # #     print("-" * 60)

# # # # ----------------------------
# # # # Save Output JSON
# # # # ----------------------------
# # # with open(output_json_path, "w") as out_file:
# # #     json.dump(results, out_file, indent=4)

# # # print(f"\n✅ Saved to: {output_json_path}")

# # import json

# # # Paths
# # input_path = "glove_good.json"
# # output_path = "TEST_VAL/notification_prompts_from_tags.json"

# # # Load album data with labels and captions
# # with open(input_path, "r") as f:
# #     data = json.load(f)

# # # Define keyword templates
# # templates = [
# #     (["wedding", "bride", "groom", "aisle", "cake"], "Looks like you had a magical wedding moment 💍 — want to create a story for it?"),
# #     (["birthday", "cake", "candles", "present"], "Birthday joy captured! 🎂 Want to relive it with a quick memory post?"),
# #     (["graduation", "diploma", "graduate", "caps"], "A proud graduation day! 🎓 Ready to make it unforgettable?"),
# #     (["sports", "trophy", "race", "hurdle", "uniforms"], "Victory and cheers! 🏆 Want to turn this sporting memory into a story?"),
# #     (["picnic", "park", "blanket", "dog", "basket"], "Lazy days and laughter 🌳🐾 — want to share this picnic moment?"),
# #     (["festival", "parade", "lanterns", "decorations"], "Festive vibes caught on camera 🎉 — want to make a story?")
# # ]

# # default_prompt = "You’ve got a beautiful moment captured — want to make a memory out of it?"

# # notification_prompts = {}

# # # Generate prompt based on original image captions (for label==1)
# # for album_id, album in data.items():
# #     if album.get("label", 0) != 1:
# #         continue  # only for meaningful albums

# #     captions = album.get("captions", [])
# #     if not captions:
# #         continue

# #     combined_text = " ".join(captions).lower()
# #     selected_prompt = default_prompt

# #     for keywords, template in templates:
# #         if any(keyword in combined_text for keyword in keywords):
# #             selected_prompt = template
# #             break

# #     notification_prompts[album_id] = {
# #         "captions": captions,
# #         "notification_prompt": selected_prompt
# #     }

# # # Save output
# # with open(output_path, "w") as f:
# #     json.dump(notification_prompts, f, indent=4, ensure_ascii=False)


# # print(f"✅ Notification prompts generated directly from image captions: {output_path}")

# import json
# import random
# import os

# # File paths
# input_json = "glove_good.json"
# output_json = "TEST_VAL/notification_prompts.json"

# # Load album data
# with open(input_json, "r", encoding="utf-8") as f:
#     data = json.load(f)

# # Define some tag-based notification templates
# tag_templates = {
#     "birthday": [
#         "Looks like a birthday bash! 🎂 Want to create a memory out of it?",
#         "Candles, cakes, and smiles! 🕯️✨ Make this birthday a story?",
#     ],
#     "wedding": [
#         "Love was in the air! 💍 Want to capture this wedding moment?",
#         "Beautiful vows and big smiles! 💒 Turn it into a story?",
#     ],
#     "sports": [
#         "Victory and cheers! 🏆 Want to turn this sporting memory into a story?",
#         "Game on and goals scored! ⚽ Ready to share this?",
#     ],
#     "graduation": [
#         "Caps flew high! 🎓 Want to remember this graduation forever?",
#         "Proud moments in gowns! 📸 Create a story for this event?",
#     ],
#     "festival": [
#         "Colors and celebrations everywhere! 🎉 Want to relive this festival?",
#         "Wasn't that festive fun? 🏮 Make a memory out of it?",
#     ],
#     "picnic": [
#         "Relaxed vibes and happy faces! 🧺 Want to save this picnic moment?",
#         "Nature, food, and laughter! 🌳 Turn this into a story?",
#     ],
#     "default": [
#         "Looks like you captured something special! 💫 Want to turn it into a memory?",
#         "This album feels meaningful. ✨ Create a story from it?",
#     ]
# }

# # Keywords to tag categories
# keywords_map = {
#     "birthday": ["birthday", "cake", "candles", "present"],
#     "wedding": ["bride", "groom", "wedding", "vows", "altar"],
#     "sports": ["trophy", "race", "uniform", "cheering", "hurdle"],
#     "graduation": ["graduate", "diploma", "cap", "graduation", "gown"],
#     "festival": ["festival", "parade", "float", "decorations", "lantern"],
#     "picnic": ["picnic", "park", "blanket", "grass", "tree", "basket"]
# }

# notification_prompts = {}

# def classify_album(captions):
#     """Simple keyword-based classifier."""
#     full_text = " ".join(captions).lower()
#     for tag, keywords in keywords_map.items():
#         if any(kw in full_text for kw in keywords):
#             return tag
#     return "default"

# # Generate prompts
# for album_id, content in data.items():
#     if content.get("label", 0) != 1:
#         continue

#     captions = content.get("captions", [])
#     if not captions:
#         continue

#     tag = classify_album(captions)
#     prompt = random.choice(tag_templates.get(tag, tag_templates["default"]))

#     notification_prompts[album_id] = {
#         "tag": tag,
#         "notification_prompt": prompt
#     }

#     print(f"Album ID: {album_id}")
#     print(f"Notification: {prompt}")
#     print("-" * 60)

# # Save to file (with proper encoding for emojis)
# with open(output_json, "w", encoding="utf-8") as f:
#     json.dump(notification_prompts, f, indent=4, ensure_ascii=False)

# print(f"✅ Saved to: {output_json}")

import json
import os
import tempfile
import shutil

# Input and output paths
input_json = "glove_good.json"
output_json = "TEST_VAL/notification_prompts.json"

# Load album data
with open(input_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

notification_prompts = {}

# Function to generate text-only notification prompt
def generate_prompt_from_tags(captions):
    captions_text = " ".join(captions).lower()

    if any(word in captions_text for word in ["birthday", "candles", "cake"]):
        return "Birthday vibes spotted! Want to create a memory?"
    elif any(word in captions_text for word in ["wedding", "bride", "groom", "aisle"]):
        return "Looks like you had fun at the wedding! Want to create a story?"
    elif any(word in captions_text for word in ["graduation", "graduate", "diploma", "cap"]):
        return "Graduation day memories! Shall we make it a post?"
    elif any(word in captions_text for word in ["sports", "trophy", "race", "hurdle", "cheering"]):
        return "Victory and cheers! Want to turn this sporting moment into a story?"
    elif any(word in captions_text for word in ["picnic", "blanket", "basket", "park", "family"]):
        return "Looks like a cozy picnic day! Want to save it as a memory?"
    elif any(word in captions_text for word in ["parade", "lanterns", "festival", "stall", "costume"]):
        return "Caught some festive spirit! Ready to share the joy?"
    else:
        return "You’ve captured a memory! Want to turn this into a story?"

# Build prompts for albums with label == 1
for album_id, item in data.items():
    if item.get("label", 0) != 1:
        continue

    captions = item.get("captions", [])
    if not captions:
        continue

    prompt = generate_prompt_from_tags(captions)

    notification_prompts[album_id] = {
        "captions": captions,
        "notification_prompt": prompt
    }

    print(f"Album ID: {album_id}")
    print(f"Notification Prompt: {prompt}")
    print("-" * 60)

# Safe JSON write function
def safe_json_write(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(filepath))
    
    with open(tmp_path, "w", encoding="utf-8") as tmp_file:
        json.dump(data, tmp_file, indent=4, ensure_ascii=False)

    os.close(tmp_fd)
    shutil.move(tmp_path, filepath)
    print(f"✅ Saved to: {filepath}")

# Save final output
safe_json_write(notification_prompts, output_json)
