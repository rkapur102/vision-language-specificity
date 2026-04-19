import random

import ijson
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# you can extract this json file from complete_data_jsons.zip, see dataset/README.md
JSON_PATH = "dataset/out_final_4_k_limited_with_scores.json"
OUTPUT_PKL = "dataset/contrast_set_size.pkl"
N_ITERS = 100
CONDITIONS = ["Image-to-Text", "Composite", "Original", "Verbose"]
CONTRAST_SIZES = [3, 10, 25, 50, 100, 250, 500, 1000, 2500]
FULL_CONTRAST_SIZE = 4999

CAPTION_TYPE_MAP = {
    "Image-to-Text": "generated_caption_4",
    "Composite": "generated_caption_2",
    "Original": "chosen_coco_caption",
    "Verbose": "generated_caption_3",
}

records = []
count = 0

with open(JSON_PATH, "r") as f:
    parser = ijson.kvitems(f, "")
    for image_id, details in parser:
        for condition in CONDITIONS:
            caption_type = CAPTION_TYPE_MAP[condition]
            clip_key = f"{caption_type}_clipscore"
            expand_key = f"{caption_type}_expanded_clipscores"

            try:
                target_score = float(details[clip_key])
            except (TypeError, ValueError, KeyError):
                continue

            expanded = details.get(expand_key) or {}
            if not expanded:
                continue

            # Exclude the current image so N means "N contrast items," with the
            # target ranked against those items plus itself.
            contrast_scores = [
                float(candidate["clipscore"])
                for candidate_id, candidate in expanded.items()
                if candidate_id != image_id
            ]
            if not contrast_scores:
                continue

            row = {
                "condition": condition,
                "image_id": image_id,
                f"rank_{FULL_CONTRAST_SIZE}": 1 + sum(
                    1 for score in contrast_scores if score > target_score
                ),
            }

            for contrast_size in CONTRAST_SIZES:
                sampled_ranks = []
                for _ in range(N_ITERS):
                    sample = random.sample(
                        contrast_scores, min(contrast_size, len(contrast_scores))
                    )
                    sampled_ranks.append(
                        1 + sum(1 for score in sample if score > target_score)
                    )
                row[f"avg_rank_{contrast_size}"] = np.mean(sampled_ranks)

            records.append(row)

        count += 1
        if count % 500 == 0:
            print(f"Processed {count} images...")

df = pd.DataFrame(records)
df.to_pickle(OUTPUT_PKL)

mean_columns = [f"avg_rank_{n}" for n in CONTRAST_SIZES] + [f"rank_{FULL_CONTRAST_SIZE}"]
summary = df.groupby("condition")[mean_columns].mean().reindex(CONDITIONS)

print(f"\nTotal rows: {len(df)}")
print("\n=== Mean rank by description type across contrast set sizes ===")
print(summary.to_string())
print(f"\nSaved to {OUTPUT_PKL}")

# Mean rank by description type across contrast set sizes:
#
#                3      10      25      50     100     250     500    1000    2500    4999
# Image-to-Text  1.004  1.014  1.034  1.069  1.139  1.345  1.693   2.391   4.456   7.911
# Composite      1.007  1.022  1.056  1.112  1.224  1.563  2.123   3.247   6.620  12.233
# Original       1.016  1.051  1.129  1.259  1.516  2.293  3.583   6.169  13.917  26.842
# Verbose        1.020  1.066  1.165  1.332  1.665  2.659  4.324   7.640  17.595  34.192
