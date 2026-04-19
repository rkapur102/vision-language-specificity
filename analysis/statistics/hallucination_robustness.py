import pandas as pd
from scipy import stats

DATAFRAME_CACHE = "dataset/hallucination_robustness.pkl"

df = pd.read_pickle(DATAFRAME_CACHE)
df = df[df["halluc_rank_1"].notna()].copy()

# Average across the three GPT-generated hallucination variants for each image.
df["avg_halluc_rank"] = df[["halluc_rank_1", "halluc_rank_2", "halluc_rank_3"]].mean(axis=1)
df["rank_increase"] = df["avg_halluc_rank"] - df["composite_rank"]

composite_ranks = df["composite_rank"].values
halluc_ranks = df["avg_halluc_rank"].values

print(f"N = {len(df)} images")
print(f"\nMean composite rank: {composite_ranks.mean():.2f}")
print(f"Mean halluc rank:    {halluc_ranks.mean():.2f}")
print(f"Mean rank increase:  {(halluc_ranks - composite_ranks).mean():.2f}")
print(
    f"\nImages where halluc rank got worse: "
    f"{(halluc_ranks > composite_ranks).sum()}/{len(df)} "
    f"({100 * (halluc_ranks > composite_ranks).mean():.1f}%)"
)
print(f"Images where halluc rank stayed same: {(halluc_ranks == composite_ranks).sum()}/{len(df)}")
print(f"Images where halluc rank improved: {(halluc_ranks < composite_ranks).sum()}/{len(df)}")

t_stat, p_value = stats.ttest_rel(halluc_ranks, composite_ranks)
print(f"\nPaired t-test: t({len(df) - 1}) = {t_stat:.3f}, p = {p_value:.6f}")

# N = 37 images
#
# Mean composite rank: 17.05
# Mean halluc rank:    30.91
# Mean rank increase:  13.86
#
# Images where halluc rank got worse: 24/37 (64.9%)
# Images where halluc rank stayed same: 10/37
# Images where halluc rank improved: 3/37
#
# Paired t-test: t(36) = 2.495, p = 0.017308
