Complete raw data (captions, etc.) can be accessed as json files at: https://drive.google.com/file/d/108wX_FTxVf0T72LWk39scAbrN3L9o8yp/view?usp=sharing. A stripped-down dataset with information needed for plotting can be found in processed_data_cache.pkl, which was created from the raw json data using create_pickle.py. The contrast-set-size ablation cache used by `analysis/statistics/contrast_set_size.py` and `analysis/statistics/contrast_set_significance.py` is stored in contrast_set_size.pkl. The canonical hallucination robustness results file is `hallucination_robustness_results.csv`, which stores the manually curated seed captions, the three generated hallucinated variants per image, their CLIPScores, and their ranks. A matching analysis cache is stored in `hallucination_robustness.pkl`.

The hallucination regeneration pipeline updates that canonical CSV in place:
1. `generation_scripts/prompt_hallucinations.py` fills the `halluc_caption_*` columns for the 37 manually verified non-hallucinated composite captions.
2. `generation_scripts/clipscores_hallucinations.py` fills the `halluc_clipscore_*` columns.
3. `compute_hallucination_ranks.py` fills the `halluc_rank_*` columns and writes `hallucination_robustness.pkl` for `analysis/statistics/hallucination_robustness.py`.

The generation_scripts folder contains files used to generate all synthetic descriptions with GPT-4o-mini as well as files used to generate CLIPscores (these were run on GPU).
