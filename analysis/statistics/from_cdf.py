import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from itertools import combinations

DATAFRAME_CACHE = "dataset/processed_data_cache.pkl"

df = pd.read_pickle(DATAFRAME_CACHE)
df["description_label"] = df["description_label"].replace({"COCO": "Original"})

conditions = ["Image-to-Text", "Composite", "Verbose", "Original"]
df_filtered = df[df["description_label"].isin(conditions)].copy()

# Within-condition: length → rank
print("\nWithin-condition: rank ~ length:\n")
for cond in conditions:
    subset = df_filtered[df_filtered['description_label'] == cond].copy()

    X_len = subset[["length"]].values
    y = subset["rank"].values
    n = len(y)

    model = LinearRegression()
    model.fit(X_len, y)

    y_pred = model.predict(X_len)
    ss_res = np.sum((y - y_pred) ** 2)
    dof = n - 2
    mse = ss_res / dof

    X_intercept = np.column_stack([np.ones(n), X_len])
    XtX_inv = np.linalg.pinv(X_intercept.T @ X_intercept)
    var_covar = mse * XtX_inv
    se = np.sqrt(np.diag(var_covar))

    coefs = np.concatenate([[model.intercept_], model.coef_])
    t_stats = coefs / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))

    beta = model.coef_[0]
    z = t_stats[1]
    p = p_values[1]

    print(f"{cond}:")
    if p < 0.05:
        sig = "p < 0.05" if p >= 0.01 else "p < 0.01" if p >= 0.001 else "p < 0.001"
        print(f"  β = {beta:.3f}, z({dof})={z:.2f}, {sig}")
    else:
        print(f"  β = {beta:.3f}, z({dof})={z:.2f}, p = {p:.4f}")

print("\nBetween-condition rank comparisons (no control):\n")

for ref, comp in combinations(conditions, 2):
    ref_ranks = df_filtered[df_filtered['description_label'] == ref]['rank'].values
    comp_ranks = df_filtered[df_filtered['description_label'] == comp]['rank'].values

    beta = comp_ranks.mean() - ref_ranks.mean()
    t_stat, p_value = stats.ttest_ind(comp_ranks, ref_ranks)
    df_resid = len(ref_ranks) + len(comp_ranks) - 2
    z = t_stat

    print(f"{comp} vs {ref}:")
    if p_value < 0.001:
        print(f"  β = {beta:.2f}, z({df_resid})={z:.2f}, p < 0.001")
    else:
        print(f"  β = {beta:.2f}, z({df_resid})={z:.2f}, p = {p_value:.4f}")

print("\nBetween-condition rank comparisons (controlling for length):\n")

for ref, comp in combinations(conditions, 2):
    df_pair = df_filtered[df_filtered['description_label'].isin([ref, comp])].copy()

    df_pair["condition_binary"] = (df_pair["description_label"] == comp).astype(int)

    y = df_pair["rank"].values
    X_full = df_pair[["condition_binary", "length"]].values
    X_length = df_pair[["length"]].values
    n = len(y)

    # Model 1: rank ~ length only
    model_length = LinearRegression()
    model_length.fit(X_length, y)
    r2_length = model_length.score(X_length, y)

    # Model 2: rank ~ condition + length
    model_full = LinearRegression()
    model_full.fit(X_full, y)
    r2_full = model_full.score(X_full, y)

    # ΔR² = incremental variance explained by condition beyond length
    delta_r2 = r2_full - r2_length

    y_pred = model_full.predict(X_full)
    ss_res = np.sum((y - y_pred) ** 2)
    dof = n - X_full.shape[1] - 1
    mse = ss_res / dof

    X_intercept = np.column_stack([np.ones(n), X_full])
    XtX_inv = np.linalg.pinv(X_intercept.T @ X_intercept)
    var_covar = mse * XtX_inv
    se = np.sqrt(np.diag(var_covar))

    coefs = np.concatenate([[model_full.intercept_], model_full.coef_])
    t_stats = coefs / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))

    beta = model_full.coef_[0]
    z = t_stats[1]
    p_value = p_values[1]

    print(f"{comp} vs {ref}:")
    if p_value < 0.001:
        print(f"  β = {beta:.2f}, z({dof})={z:.2f}, p < 0.001, ΔR² = {delta_r2:.4f}")
    else:
        print(f"  β = {beta:.2f}, z({dof})={z:.2f}, p = {p_value:.4f}, ΔR² = {delta_r2:.4f}")

print("\nBetween-condition rank comparisons (controlling for image, mixed model):\n")

import statsmodels.formula.api as smf

for ref, comp in combinations(conditions, 2):
    df_pair = df_filtered[df_filtered['description_label'].isin([ref, comp])].copy()
    df_pair["condition_binary"] = (df_pair["description_label"] == comp).astype(int)

    md = smf.mixedlm("rank ~ condition_binary", df_pair, groups=df_pair["image_id"])
    mdf = md.fit(reml=True)

    beta = mdf.fe_params["condition_binary"]
    z = mdf.tvalues["condition_binary"]
    p_value = mdf.pvalues["condition_binary"]
    n = len(df_pair)
    n_groups = len(df_pair["image_id"].unique())

    print(f"{comp} vs {ref} (n={n}, n_images={n_groups}):")
    if p_value < 0.001:
        print(f"  β = {beta:.2f}, z={z:.2f}, p < 0.001")
    else:
        print(f"  β = {beta:.2f}, z={z:.2f}, p = {p_value:.4f}")

print("\nBetween-condition length comparisons:\n")

for ref, comp in combinations(conditions, 2):
    ref_lengths = df_filtered[df_filtered['description_label'] == ref]['length'].values
    comp_lengths = df_filtered[df_filtered['description_label'] == comp]['length'].values

    beta = comp_lengths.mean() - ref_lengths.mean()
    t_stat, p_value = stats.ttest_ind(comp_lengths, ref_lengths)
    df_resid = len(ref_lengths) + len(comp_lengths) - 2
    z = t_stat

    print(f"{comp} vs {ref}:")
    if p_value < 0.001:
        print(f"  β = {beta:.2f}, z({df_resid})={z:.2f}, p < 0.001")
    else:
        print(f"  β = {beta:.2f}, z({df_resid})={z:.2f}, p = {p_value:.4f}")

# Within-condition: rank ~ length:

# Image-to-Text:
#   β = 0.010, z(941)=0.58, p = 0.5647
# Composite:
#   β = 0.021, z(3662)=2.86, p < 0.01
# Verbose:
#   β = -0.024, z(4330)=-1.50, p = 0.1342
# Original:
#   β = -0.222, z(4998)=-2.42, p < 0.05

# Between-condition comparisons (no control):

# Composite vs Image-to-Text:
#   β = 4.32, z(4605)=4.15, p < 0.001
# Verbose vs Image-to-Text:
#   β = 26.28, z(5273)=8.68, p < 0.001
# Original vs Image-to-Text:
#   β = 18.93, z(5941)=7.11, p < 0.001
# Verbose vs Composite:
#   β = 21.96, z(7994)=13.72, p < 0.001
# Original vs Composite:
#   β = 14.61, z(8662)=10.33, p < 0.001
# Original vs Verbose:
#   β = -7.35, z(9330)=-4.08, p < 0.001

# Between-condition comparisons (controlling for length):

# Composite vs Image-to-Text:
#   β = 5.58, z(4604)=4.99, p < 0.001, ΔR² = 0.0054
# Verbose vs Image-to-Text:
#   β = 23.29, z(5272)=6.56, p < 0.001, ΔR² = 0.0080
# Original vs Image-to-Text:
#   β = -8.28, z(5940)=-0.50, p = 0.6140, ΔR² = 0.0000
# Verbose vs Composite:
#   β = 21.35, z(7993)=12.24, p < 0.001, ΔR² = 0.0183
# Original vs Composite:
#   β = 16.97, z(8661)=4.60, p < 0.001, ΔR² = 0.0024
# Original vs Verbose:
#   β = -11.65, z(9329)=-4.05, p < 0.001, ΔR² = 0.0018

# Between-condition length comparisons:

# Composite vs Image-to-Text:
#   β = -60.97, z(4605)=-26.56, p < 0.001
# Verbose vs Image-to-Text:
#   β = -131.03, z(5273)=-44.64, p < 0.001
# Original vs Image-to-Text:
#   β = -283.81, z(5941)=-469.03, p < 0.001
# Verbose vs Composite:
#   β = -70.05, z(7994)=-38.88, p < 0.001
# Original vs Composite:
#   β = -222.84, z(8662)=-224.43, p < 0.001
# Original vs Verbose:
#   β = -152.79, z(9330)=-120.12, p < 0.001
