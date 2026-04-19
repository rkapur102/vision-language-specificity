import time
import pandas as pd
from openai import OpenAI

# removed api key 
# os.environ['OPENAI_API_KEY'] =

client = OpenAI()

# This script updates the canonical hallucination results CSV in place. The
# starting file should already contain the manually curated composite captions
# plus the `hallucination` indicator used to select the 37 non-hallucinated seeds.
csv_path = '../hallucination_robustness_results.csv'

df = pd.read_csv(csv_path)

# filter to only non-hallucinated composite captions
no_halluc = df[df['hallucination'] == 0].copy()

print(f"Processing {len(no_halluc)} non-hallucinated captions")

# initialize new columns if they don't exist
for col in ['halluc_caption_1', 'halluc_caption_2', 'halluc_caption_3']:
    if col not in df.columns:
        df[col] = None


def generate_one_hallucination(composite_caption, retries = 3, wait_time = 60):
    attempt = 0
    while attempt < retries:
        try:
            prompt = (
                f"Change one detail in this description so that it becomes "
                f"incompatible with the original."
                f"Only output the changed description and nothing else.\n\n"
                f"{composite_caption}"
            )

            response = client.chat.completions.create(
                model = "gpt-4o-mini",
                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            attempt += 1
            print(f"Error occurred: {e}. Attempt {attempt} of {retries}. Waiting for {wait_time} seconds before retrying.")
            time.sleep(wait_time)
            if attempt == retries:
                print("Max retries reached. Skipping this request.")
                return None


num = 0
for idx, row in no_halluc.iterrows():
    composite_caption = row['composite_caption']
    image_id = row['image_id']

    print(f"[{num+1}/{len(no_halluc)}] {image_id}")

    for i in range(1, 4):
        col = f'halluc_caption_{i}'
        result = generate_one_hallucination(composite_caption)
        df.at[idx, col] = result
        print(f"  {i}: {result}")

    num += 1

    if num % 10 == 0:
        df.to_csv(csv_path, index = False)
        print(f"Progress saved after {num} captions.")

df.to_csv(csv_path, index = False)
print(f"\nDone. Saved to {csv_path}")
