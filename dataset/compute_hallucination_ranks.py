import ijson
import pandas as pd

CSV_PATH = "dataset/hallucination_robustness_results.csv"
PKL_PATH = "dataset/hallucination_robustness.pkl"
# you can extract this json file from complete_data_jsons.zip, see dataset/README.md
JSON_PATH = "dataset/out_final_4_k_limited_with_scores.json"

df = pd.read_csv(CSV_PATH)
has_scores = df[df["halluc_clipscore_1"].notna()]

# Only images with scored hallucinated captions need expanded CLIPScores.
needed_ids = set(has_scores["image_id"].values)

expanded_scores = {}
with open(JSON_PATH, "r") as f:
    parser = ijson.kvitems(f, "")
    for image_id, details in parser:
        if image_id in needed_ids:
            expanded = details.get("generated_caption_2_expanded_clipscores") or {}
            expanded_scores[image_id] = [float(value["clipscore"]) for value in expanded.values()]
            if len(expanded_scores) == len(needed_ids):
                break

print(f"Loaded expanded scores for {len(expanded_scores)} images")

for hallucination_idx in range(1, 4):
    rank_col = f"halluc_rank_{hallucination_idx}"
    if rank_col not in df.columns:
        df[rank_col] = None

for idx, row in has_scores.iterrows():
    image_id = row["image_id"]
    all_scores = expanded_scores.get(image_id, [])
    if not all_scores:
        print(f"WARNING: no expanded scores for {image_id}")
        continue

    for hallucination_idx in range(1, 4):
        halluc_score = row[f"halluc_clipscore_{hallucination_idx}"]
        rank = 1 + sum(1 for score in all_scores if score > halluc_score)
        df.at[idx, f"halluc_rank_{hallucination_idx}"] = rank

    print(
        f"{image_id}: composite_rank={int(row['composite_rank'])}, "
        f"halluc_ranks=[{int(df.at[idx, 'halluc_rank_1'])}, "
        f"{int(df.at[idx, 'halluc_rank_2'])}, {int(df.at[idx, 'halluc_rank_3'])}]"
    )

df.to_csv(CSV_PATH, index=False)
df.to_pickle(PKL_PATH)
print(f"\nSaved to {CSV_PATH} and {PKL_PATH}")
