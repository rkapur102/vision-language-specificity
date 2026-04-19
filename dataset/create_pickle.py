import ijson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import torch
import clip
import warnings
import os

rcParams['pdf.fonttype'] = 42
sns.set_style("white")

# you can extract these json files from complete_data_jsons.zip, see dataset/README.md
FILENAME_1 = "out_final_4_k_limited_with_scores.json"
FILENAME_2 = "OF4_gpt4o_updated.json"
DATAFRAME_CACHE = "processed_data_cache.pkl"

caption_types_file1 = [
    "generated_caption_4", # image-to-text
    "chosen_coco_caption", # COCO
    "generated_caption_4_k_limited", # k-Limited
    "generated_caption_2", # composite
    "generated_caption_3", # verbose
]

caption_types_file2 = [
    "gpt4o_concise", # concise
    "gpt4o_200char", # character-limited
]

label_map = {
    "generated_caption_4": "Image-to-Text",
    "chosen_coco_caption": "COCO",
    "generated_caption_4_k_limited": "k-Limited",
    "generated_caption_2": "Composite",
    "generated_caption_3": "Verbose",
    "gpt4o_concise": "Concise",
    "gpt4o_200char": "200-Character-Limited",
}

label_color_map = {
    "Image-to-Text": "#785EF0",
    "COCO": "#FE6100",
    "k-Limited": "#648FFF",
    "Composite": "#DC267F",
    "Verbose": "#FFB000",
    "Concise": "#44AA99",
    "200-Character-Limited": "#B6C45E",
}

length_bins = list(range(25, 426, 25)) + [float("inf")]
length_labels = [f"{start}-{end}" for start, end in zip(length_bins[:-2], length_bins[1:-1])] + ["425+"]

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == 'cpu':
    warnings.warn("Running on CPU — results may differ from float16 GPU results.")

model, _ = clip.load("ViT-B/32", device=device)
model.eval()

records = []
coco_captions_map = {}

# first file
with open(FILENAME_1, "r") as f:
    parser = ijson.kvitems(f, "")
    count = 0

    for image_id, details in parser:
        captions_list = details.get("captions", [])
        if captions_list:
            coco_captions_map[image_id] = captions_list

        for cap_type in caption_types_file1:
            caption = details.get(cap_type)
            if caption is None or caption.strip() == "":
                continue
            try:
                tokenized_text = clip.tokenize([caption])
                token_count = tokenized_text.shape[1]
            except Exception:
                token_count = 78
            if token_count > 77:
                continue

            clip_key = f"{cap_type}_clipscore"
            expand_key = f"{cap_type}_expanded_clipscores"
            length_key = f"{cap_type}_length"

            if clip_key not in details or expand_key not in details:
                continue
            try:
                real_val = float(details[clip_key])
            except (TypeError, ValueError):
                continue

            expanded = details[expand_key] or {}
            rank = 1 + sum(1 for v in expanded.values() if float(v["clipscore"]) > real_val)

            try:
                length_val = float(details.get(length_key, np.nan))
            except ValueError:
                length_val = np.nan

            records.append({
                "image_id": image_id,
                "caption": caption,
                "description_label": label_map[cap_type],
                "rank": rank,
                "length": length_val,
            })

        count += 1
        if count % 1000 == 0:
            print(f"  Processed {count} images...")

# second file
with open(FILENAME_2, "r") as f:
    parser = ijson.kvitems(f, "")
    count = 0
    
    for image_id, details in parser:
        for cap_type in caption_types_file2:
            caption = details.get(cap_type)
            if caption is None or caption.strip() == "":
                continue
            try:
                tokenized_text = clip.tokenize([caption])
                token_count = tokenized_text.shape[1]
            except Exception:
                token_count = 78
            if token_count > 77:
                continue

            clip_key = f"{cap_type}_clipscore"
            expand_key = f"{cap_type}_expanded_clipscores"
            length_key = f"{cap_type}_length"

            if clip_key not in details or expand_key not in details:
                continue
            try:
                real_val = float(details[clip_key])
            except (TypeError, ValueError):
                continue

            expanded = details[expand_key] or {}
            if not expanded:
                continue
            rank = 1 + sum(1 for v in expanded.values() if float(v["clipscore"]) > real_val)

            try:
                length_val = float(details.get(length_key, np.nan))
            except ValueError:
                length_val = np.nan

            records.append({
                "image_id": image_id,
                "caption": caption,
                "description_label": label_map[cap_type],
                "rank": rank,
                "length": length_val,
            })

        count += 1
        if count % 1000 == 0:
            print(f"  Processed {count} images...")

df = pd.DataFrame(records)

coco_df = pd.DataFrame([
    {"image_id": img_id, **{f"coco_caption_{i+1}": cap for i, cap in enumerate(caps)}}
    for img_id, caps in coco_captions_map.items()
])
df = df.merge(coco_df, on="image_id", how="left")

df.to_pickle(DATAFRAME_CACHE)