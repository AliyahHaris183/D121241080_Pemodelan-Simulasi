import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, shapiro

# ==========================================
# DATA RATING FILM LETTERBOXD
# ==========================================

rating = [
    4.1, 4.2, 2.8, 3.0, 3.5, 2.5, 2.6, 3.5, 3.8, 3.4,
    4.2, 3.9, 3.5, 3.4, 3.6, 4.4, 3.8, 4.2, 4.2, 4.4,
    3.9, 3.8, 4.5, 3.8, 4.4, 3.2, 3.9, 4.5, 4.1, 4.4,
    4.0, 3.7, 4.0, 3.3, 3.2, 4.3, 3.8, 4.4, 3.0, 3.4
]

# Membuat DataFrame
df = pd.DataFrame({
    "Rating": rating
})


# ==========================================
# STATISTIK DASAR
# ==========================================

mean = df["Rating"].mean()
median = df["Rating"].median()
std = df["Rating"].std()
skewness = skew(df["Rating"])

print("==========================================")
print("     STATISTIK DATA RATING FILM")
print("==========================================")

print(f"Jumlah data       : {len(rating)}")
print(f"Mean              : {mean:.2f}")
print(f"Median            : {median:.2f}")
print(f"Standar deviasi   : {std:.2f}")
print(f"Skewness          : {skewness:.2f}")


# ==========================================
# UJI NORMALITAS SHAPIRO-WILK
# ==========================================

stat, p_value = shapiro(df["Rating"])

print("\n==========================================")
print("       UJI NORMALITAS SHAPIRO-WILK")
print("==========================================")

print(f"Statistik         : {stat:.4f}")
print(f"p-value           : {p_value:.4f}")

if p_value > 0.05:
    print("Kesimpulan        : Data tidak menunjukkan")
    print("                    penyimpangan signifikan")
    print("                    dari distribusi normal.")
else:
    print("Kesimpulan        : Data menunjukkan")
    print("                    penyimpangan signifikan")
    print("                    dari distribusi normal.")


# ==========================================
# INTERPRETASI SKEWNESS
# ==========================================

print("\n==========================================")
print("          INTERPRETASI SKEWNESS")
print("==========================================")

if skewness > 0:
    print("Distribusi cenderung menceng ke kanan")
    print("(positively skewed).")

elif skewness < 0:
    print("Distribusi cenderung menceng ke kiri")
    print("(negatively skewed).")

else:
    print("Distribusi relatif simetris.")


# ==========================================
# HISTOGRAM
# ==========================================

plt.figure(figsize=(8, 5))

plt.hist(
    df["Rating"],
    bins=np.arange(2.45, 4.56, 0.2),
    color="#AEFEF6",
    edgecolor="#000000",
    linewidth=1.2
)

plt.title(
    "Distribusi Rating Film pada Dataset Letterboxd",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Rating", fontsize=11)
plt.ylabel("Frekuensi", fontsize=11)

plt.xticks(np.arange(2.5, 4.6, 0.2))

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()
