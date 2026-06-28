import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/results/localizations.csv")

plt.rcParams.update({"font.family": "serif", "font.size": 11,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.dpi": 180, "savefig.facecolor": "white"})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(df["crack_area_pct"] * 100, bins=40, color="#2E75B6", edgecolor="#0F4070", alpha=0.85)
axes[0].axvline(df["crack_area_pct"].mean() * 100, color="#E67E22", linestyle="--", linewidth=2,
                label=f"Mean = {df['crack_area_pct'].mean()*100:.2f}%")
axes[0].axvline(df["crack_area_pct"].median() * 100, color="#70AD47", linestyle=":", linewidth=2,
                label=f"Median = {df['crack_area_pct'].median()*100:.2f}%")
axes[0].set_xlabel("Crack area (% of image)")
axes[0].set_ylabel("Number of detections")
axes[0].set_title("(a) Crack area distribution")
axes[0].legend(loc="upper right", fontsize=10)
axes[0].grid(alpha=0.3); axes[0].set_axisbelow(True)

axes[1].hist(df["crack_length_px"], bins=40, color="#ED7D31", edgecolor="#A14B0F", alpha=0.85)
axes[1].axvline(df["crack_length_px"].mean(), color="#2E75B6", linestyle="--", linewidth=2,
                label=f"Mean = {df['crack_length_px'].mean():.0f} px")
axes[1].axvline(df["crack_length_px"].median(), color="#70AD47", linestyle=":", linewidth=2,
                label=f"Median = {df['crack_length_px'].median():.0f} px")
axes[1].set_xlabel("Skeleton length (pixels)")
axes[1].set_ylabel("Number of detections")
axes[1].set_title("(b) Crack length distribution")
axes[1].legend(loc="upper right", fontsize=10)
axes[1].grid(alpha=0.3); axes[1].set_axisbelow(True)

plt.suptitle("Distribution of crack area and length across 9,149 detections on SDNET2018",
             fontsize=14, fontweight="bold", y=1.04)
plt.savefig("data/results/fig_4_5_distributions_real.png", dpi=180)
print("Saved data/results/fig_4_5_distributions_real.png")
