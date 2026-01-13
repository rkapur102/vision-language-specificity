# When More Words Say Less: Decoupling Length and Informativity in Image Description Evaluation

This is the official repository for Kapur et al. (2026): "When More Words Say Less: Decoupling Length and Informativity in Image Description Evaluation" (preprint: https://arxiv.org/abs/2601.04609).

## Repository structure

### `dataset/`

This folder contains the synthetic caption dataset generated from COCO images. Complete raw data (captions, image paths, CLIPScores, lengths, additional metadata) is available as JSON files at: https://drive.google.com/file/d/108wX_FTxVf0T72LWk39scAbrN3L9o8yp/view?usp=sharing

A processed version with length and rank data is provided in `processed_data_cache.pkl`, created using `create_pickle.py`.

The `generation_scripts/` subdirectory contains code for generating synthetic descriptions with GPT-4o-mini and computing CLIPScores.

### `analysis/`

This folder contains scripts for generating figures and statistics reported in the paper:
- `figures/` — generates Figures 2, 3, and 4
- `statistics/` — generates significance tests from Sections 4.2/4.3 (`from_cdf.py`) and 4.4 (`from_vlm.py`)

### `human_subject_experiment/`

This folder contains two human subject studies: `informativity_study/` (informativity task) and `preference_study/` (preference task, main study).

To run either study, open `index.html` in your browser.

- `data/` — anonymized participant data
- `analysis/` — R Markdown file with all analyses and statistical tests

## Contact

Email rheak@stanford.edu with any inquiries.
