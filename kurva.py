import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ==========================================
# DATA
# ==========================================
rating = [
    4.1, 4.2, 2.8, 3.0, 3.5, 2.5, 2.6, 3.5, 3.8, 3.4,
    4.2, 3.9, 3.5, 3.4, 3.6, 4.4, 3.8, 4.2, 4.2, 4.4,
    3.9, 3.8, 4.5, 3.8, 4.4, 3.2, 3.9, 4.5, 4.1, 4.4,
    4.0, 3.7, 4.0, 3.3, 3.2, 4.3, 3.8, 4.4, 3.0, 3.4,
]
data = np.array(sorted(rating))
n = len(data)

# ==========================================
# FITTING BETA (MLE), skala domain [0.5, 5.0]
# ==========================================
lo, hi = 0.5, 5.0
rng = hi - lo
scaled = (data - lo) / rng
a_hat, b_hat, _, _ = stats.beta.fit(scaled, floc=0, fscale=1)

x = np.linspace(lo, hi, 500)
y = (x - lo) / rng
pdf_beta = stats.beta.pdf(y, a_hat, b_hat) / rng
cdf_beta = stats.beta.cdf(y, a_hat, b_hat)

ecdf_x = data
ecdf_y = np.arange(1, n + 1) / n

fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))

# ---------------- PDF ----------------
ax = axes[0]
ax.hist(data, bins=np.arange(2.5, 4.71, 0.2), density=True, color="#AEE9FE",
        edgecolor="black", linewidth=1, alpha=0.85, label="Histogram (data)")
ax.plot(x, pdf_beta, color="#1f77b4", lw=2.5, label=f"PDF Beta (a={a_hat:.2f}, b={b_hat:.2f})")
ax.axvline(lo + (a_hat-1)/(a_hat+b_hat-2)*rng, color="grey", ls="--", lw=1, alpha=0.7)
ax.set_title("Kurva PDF Distribusi Beta", fontsize=13, fontweight="bold")
ax.set_xlabel("Rating (x)")
ax.set_ylabel("Densitas Probabilitas f(x)")
ax.set_xlim(2.2, 4.9)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)

eq_pdf = (
    r"$f(x)=\frac{1}{4.5}\cdot\frac{y^{9.4747}(1-y)^{2.9749}}{B(10.4747,\ 3.9749)}$" "\n"
    r"$y=\frac{x-0.5}{4.5},\ \ x\in[0.5,\ 5.0]$"
)
ax.text(0.98, 0.02, eq_pdf, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10.5, bbox=dict(boxstyle="round,pad=0.5", fc="#f5f7fa", ec="#9aa5b1"))

# ---------------- CDF ----------------
ax = axes[1]
ax.step(ecdf_x, ecdf_y, where="post", color="black", lw=1.5, label="CDF Empiris (data)")
ax.plot(x, cdf_beta, color="#1f77b4", lw=2.5, label="CDF Beta")
ax.set_title("Kurva CDF Distribusi Beta", fontsize=13, fontweight="bold")
ax.set_xlabel("Rating (x)")
ax.set_ylabel("Probabilitas Kumulatif F(x)")
ax.set_xlim(2.2, 4.9)
ax.set_ylim(0, 1.02)
ax.legend(fontsize=9.5, loc="upper left")
ax.grid(alpha=0.3)

eq_cdf = (
    r"$F(x)=I_y(10.4747,\ 3.9749)=\frac{1}{B(a,b)}\int_0^{y} t^{a-1}(1-t)^{b-1}\,dt$" "\n"
    r"$y=\frac{x-0.5}{4.5}$"
)
ax.text(0.98, 0.02, eq_cdf, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="#f5f7fa", ec="#9aa5b1"))

plt.tight_layout()

plt.savefig("beta_pdf_cdf.png", dpi=150)
print("saved successfully to beta_pdf_cdf.png")

plt.show()
