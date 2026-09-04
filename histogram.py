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


# ==========================================
# DATAFRAME
# ==========================================

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
print("       STATISTIK DATA RATING FILM")
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
print("      UJI NORMALITAS SHAPIRO-WILK")
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

plt.figure(figsize=(10, 5))

# Lebar kelas = 0.2

# Data terendah = 2.5
# Data tertinggi = 4.5

bins = np.arange(2.4, 4.61, 0.2)


plt.hist(
    rating,
    bins=bins,
    color="#AEFEF6",
    edgecolor="black",
    linewidth=1.2
)


# ==========================================
# JUDUL DAN LABEL
# ==========================================

plt.title(
    "Distribusi Rating Film pada Dataset Letterboxd",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel(
    "Rating",
    fontsize=11
)

plt.ylabel(
    "Frekuensi",
    fontsize=11
)


# ==========================================
# TANDA PADA SUMBU X
# ==========================================

plt.xticks(
    [2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0, 4.2, 4.5]
)


# ==========================================
# JARAK GRAFIK
# ==========================================

plt.xlim(
    2.35,
    4.65
)


# ==========================================
# GRID
# ==========================================

plt.grid(
    axis="y",
    alpha=0.3
)


# ==========================================
# TAMPILKAN
# ==========================================

plt.tight_layout()
plt.show()
