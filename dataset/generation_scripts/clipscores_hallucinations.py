'''
MOST OF THIS CODE IS EITHER COPIED EXACTLY FROM OR ADAPTED FROM EITHER 1) : 
Code for CLIPScore (https://arxiv.org/abs/2104.08718)
@inproceedings{hessel2021clipscore,
  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
  booktitle={EMNLP},
  year={2021}
}
AND/OR 2) : https://github.com/elisakreiss/contextual-description-evaluation/blob/main/metrics/clipscore/clipscore.py
'''

import argparse
import clip
import torch
import ijson
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import tqdm
import numpy as np
import sklearn.preprocessing
import collections
import os
import pathlib
import json
import random
import generation_eval_utils
import pprint
import warnings
from packaging import version
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--shuffle',
        type = bool,
        default = False,
        help = 'Optional: shuffled image--text pairs')

    parser.add_argument(
        '--references_json',
        default = None,
        help = 'Optional references json mapping from image_id --> [list of references]')

    parser.add_argument(
        '--compute_other_ref_metrics',
        default = 1,
        type = int,
        help = 'If references is specified, should we compute standard reference-based metrics?')

    parser.add_argument(
        '--save_per_instance',
        default = None,
        help = 'if set, we will save per instance clipscores to this file')

    args = parser.parse_args()

    if isinstance(args.save_per_instance, str) and not args.save_per_instance.endswith('.json'):
        print('if you\'re saving per-instance, please make sure the filepath ends in json.')
        quit()
    return args


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix = 'A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate = True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation = Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size = 256, num_workers = 16):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size = batch_size, num_workers = num_workers, shuffle = False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, device, batch_size = 64, num_workers = 16):
    print("extract_all_images len(images): ", len(images))
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size = batch_size, num_workers = num_workers, shuffle = False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['image'].to(device)
            if device == 'cuda':
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def get_clip_score(model, images, candidates, device, w = 2.5):
    if isinstance(images, list):
        images = extract_all_images(images, model, device)
    candidates = extract_all_captions(candidates, model, device)

    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis = 1)
        candidates = sklearn.preprocessing.normalize(candidates, axis = 1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images**2, axis = 1, keepdims = True))
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis = 1, keepdims = True))
    per = w * np.clip(np.sum(images * candidates, axis = 1), 0, None)
    print(per)
    result_list = per.tolist()
    return result_list


def get_refonlyclipscore(model, references, candidates, device):
    if isinstance(candidates, list):
        candidates = extract_all_captions(candidates, model, device)

    flattened_refs = []
    flattened_refs_idxs = []
    for idx, refs in enumerate(references):
        flattened_refs.extend(refs)
        flattened_refs_idxs.extend([idx for _ in refs])

    flattened_refs = extract_all_captions(flattened_refs, model, device)

    if version.parse(np.__version__) < version.parse('1.21'):
        candidates = sklearn.preprocessing.normalize(candidates, axis = 1)
        flattened_refs = sklearn.preprocessing.normalize(flattened_refs, axis = 1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis = 1, keepdims = True))
        flattened_refs = flattened_refs / np.sqrt(np.sum(flattened_refs**2, axis = 1, keepdims = True))

    cand_idx2refs = collections.defaultdict(list)
    for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feats)

    assert len(cand_idx2refs) == len(candidates)

    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

    per = []
    for c_idx, cand in tqdm.tqdm(enumerate(candidates)):
        cur_refs = cand_idx2refs[c_idx]
        all_sims = cand.dot(cur_refs.transpose())
        per.append(np.max(all_sims))

    return np.mean(per), per


def main():
    # This script fills `halluc_clipscore_*` columns in the canonical
    # hallucination results CSV after hallucinated variants have been generated.
    csv_path = '../hallucination_robustness_results.csv'
    output_csv = '../hallucination_robustness_results.csv'

    df = pd.read_csv(csv_path)
    has_halluc = df[df['halluc_caption_1'].notna()]

    # collect all image paths and captions, aligned so each caption matches its image
    image_paths = []
    captions = []
    caption_indices = []  # (df_idx, halluc_num) to map scores back

    for idx, row in has_halluc.iterrows():
        image_id = row['image_id']
        image_path = f'./coco_dataset/train2017/{image_id}'

        for h in range(1, 4):
            caption = row[f'halluc_caption_{h}']
            if pd.isna(caption) or caption == "":
                continue

            try:
                tokenized_text = clip.tokenize(caption)
                token_count = tokenized_text.shape[1]
            except Exception:
                token_count = 78

            if token_count <= 77:
                captions.append(caption)
            else:
                captions.append("")

            image_paths.append(image_path)
            caption_indices.append((idx, h))

    print(f"Computing CLIPScores for {len(captions)} hallucinated captions...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn("Running on CPU - results may differ from float16 GPU results.")

    model, _ = clip.load("ViT-B/32", device = device)
    model.eval()

    # extract features for all images and captions
    image_feats = extract_all_images(image_paths, model, device)
    scores = get_clip_score(model, image_feats, captions, device)

    # initialize new columns
    for h in range(1, 4):
        if f'halluc_clipscore_{h}' not in df.columns:
            df[f'halluc_clipscore_{h}'] = None

    # map scores back to dataframe
    for (df_idx, h), score in zip(caption_indices, scores):
        df.at[df_idx, f'halluc_clipscore_{h}'] = score

    df.to_csv(output_csv, index = False)
    print(f"\nDone. Saved CLIPScores to {output_csv}")

    # print summary
    for idx, row in df[df['halluc_clipscore_1'].notna()].iterrows():
        print(f"{row['image_id']}: composite_rank={row['composite_rank']}, "
              f"halluc_scores=[{row['halluc_clipscore_1']:.4f}, {row['halluc_clipscore_2']:.4f}, {row['halluc_clipscore_3']:.4f}]")


if __name__ == "__main__":
    main()
