import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

# ## Original time REBOUND integration
ot = pd.read_parquet("data/original_time.pq")

# ## Triple time REBOUND integration
tt = pd.read_parquet("data/triple_time.pq")

## Triple time STARFORGE data
stt = pd.read_parquet("data/end_state_all_time3.pq")

nt = pd.read_parquet("data/newtonian_trials.pq")

# ## Compute ratios
ratio_ao = ot["a_rebound"] / ot["a_expected"]
ratio_eo = ot["e_rebound"] / ot["e_expected"]

# ratio_at = tt["a_rebound"] / tt["a_expected"]
# ratio_et = tt["e_rebound"] / tt["e_expected"]

ratio_att = tt["a_rebound"] / stt["sma"]
ratio_ett = tt["e_rebound"] / stt["ecc"]

# ## KDEs for semimajor axis
# kde_ao = gaussian_kde(ratio_ao)
# kde_at = gaussian_kde(ratio_at)
# x_vals_a = np.linspace(min(min(ratio_ao),min(ratio_at)), max(max(ratio_ao),max(ratio_at)), 1000)

# pdf_ao = kde_ao(x_vals_a)
# pdf_at = kde_at(x_vals_a)

# ## KDEs for eccentricity
# kde_eo = gaussian_kde(ratio_eo)
# kde_et = gaussian_kde(ratio_et)

# x_vals_e = np.linspace(min(min(ratio_eo),min(ratio_et)), max(max(ratio_eo),max(ratio_et)), 1000)

# pdf_eo = kde_eo(x_vals_e)
# pdf_et = kde_et(x_vals_e)
 

# # Histogram or PDF plot
# def make_overlay_plot(data1, data2, pdf1, pdf2, x_vals, label, plot_type, filename):
# 	fig, ax = plt.subplots(figsize=(6, 4))
# 	all_data = np.concatenate([data1, data2])
# 	bins = np.histogram_bin_edges(ot["e_rebound"], bins="fd", range=None, weights=None)

# 	if plot_type == "unnormalized":
# 		ax.hist(data1, bins=bins, histtype="stepfilled", edgecolor='black', color='skyblue', alpha=0.5, label="Original Time")
# 		ax.hist(data2, bins=bins, histtype="stepfilled", edgecolor='black', color='orange', alpha=0.5, label="Triple Time")
# 		ax.set_ylabel("Count")
# 		ax.set_title(f"Unnormalized Histogram ({label})")
# 	elif plot_type == "pdf":
# 		ax.hist(data1, bins=bins, histtype="stepfilled", density=True, edgecolor='black', color='skyblue', alpha=0.3, label="Original Time (hist)")
# 		ax.hist(data2, bins=bins, histtype="stepfilled", density=True, edgecolor='black', color='orange', alpha=0.3, label="Triple Time (hist)")
# 		ax.set_xlim(0,3)
# 		ax.plot(x_vals, pdf1, color='blue', linewidth=2, label="Original Time (KDE)")
# 		ax.plot(x_vals, pdf2, color='darkorange', linewidth=2, label="Triple Time (KDE)")
# 		ax.set_ylabel("Probability Density")
# 		ax.set_title(f"PDF + Histogram ({label})")

# 	ax.axvline(1, color='red', linestyle='--', label='1:1 Match')
# 	ax.set_xlabel(f"Rebound / Expected ({label})")
# 	ax.legend()
# 	ax.grid(True)
# 	fig.tight_layout()
# 	fig.savefig(filename)
# 	plt.close(fig)

# # # ECDF with connected line
# def plot_ecdf_overlay(data1, data2, label, filename):
# 	x1 = np.sort(data1)
# 	y1 = np.arange(1, len(x1)+1) / len(x1)

# 	x2 = np.sort(data2)
# 	y2 = np.arange(1, len(x2)+1) / len(x2)

# 	fig, ax = plt.subplots(figsize=(6, 4))
# 	ax.plot(x1, y1, label="Original Time", linestyle='-', color='blue')
# 	ax.plot(x2, y2, label="Triple Time", linestyle='--', color='darkorange')
# 	ax.axvline(1, color='red', linestyle='--', label='1:1 Match')
# 	ax.set_title(f"ECDF Comparison ({label})")
# 	ax.set_xlabel(f"Rebound / Expected ({label})")
# 	ax.set_ylabel("Cumulative Probability")
# 	ax.set_xlim(0,6)
# 	ax.legend()
# 	ax.grid(True)
# 	fig.tight_layout()
# 	fig.savefig(filename)
# 	plt.close(fig)


#################################################################################

## Filtering infinite and non-numbers
maska = (~np.isnan(ratio_att)) & (~np.isinf(ratio_att))
maske = (~np.isnan(ratio_ett)) & (~np.isinf(ratio_ett))

ratio_att = ratio_att[maska]
ratio_ett = ratio_ett[maske]

## KDEs for semimajor axis
kde_ao = gaussian_kde(ratio_ao)
kde_at = gaussian_kde(ratio_att)
x_vals_a = np.linspace(min(min(ratio_ao),min(ratio_att)), max(max(ratio_ao),max(ratio_att)), 1000)

pdf_ao = kde_ao(x_vals_a)
pdf_at = kde_at(x_vals_a)

## KDEs for eccentricity
kde_eo = gaussian_kde(ratio_eo)
kde_et = gaussian_kde(ratio_ett)

x_vals_e = np.linspace(min(min(ratio_eo),min(ratio_ett)), max(max(ratio_eo),max(ratio_ett)), 1000)

pdf_eo = kde_eo(x_vals_e)
pdf_et = kde_et(x_vals_e)


# Histogram or PDF plot
def make_overlay_plot(data1, data2, pdf1, pdf2, x_vals, label, plot_type, filename, xlim=None):
	fig, ax = plt.subplots(figsize=(6, 4))
	all_data = np.concatenate([data1, data2])
	bins = np.histogram_bin_edges(tt["e_rebound"], bins="fd", range=None, weights=None)
	if xlim:
			ax.set_xlim(xlim)

	if plot_type == "unnormalized":
		ax.hist(data1, bins=bins, histtype="stepfilled", edgecolor='black', color='skyblue', alpha=0.5, label="Original Time")
		ax.hist(data2, bins=bins, histtype="stepfilled", edgecolor='black', color='orange', alpha=0.5, label="Triple Time")
		ax.set_ylabel("Count")
		ax.set_title(f"Unnormalized Histogram ({label}) Triple Time")
	elif plot_type == "pdf":
		ax.hist(data1, bins=bins, histtype="stepfilled", density=True, edgecolor='black', color='skyblue', alpha=0.3, label="Original Time (hist)")
		ax.hist(data2, bins=bins, histtype="stepfilled", density=True, edgecolor='black', color='orange', alpha=0.3, label="Triple Time (hist)")
		ax.plot(x_vals, pdf1, color='blue', linewidth=2, label="Original Time (KDE)")
		ax.plot(x_vals, pdf2, color='darkorange', linewidth=2, label="Triple Time (KDE)")
		ax.set_ylabel("Probability Density")
		ax.set_title(f"PDF + Histogram ({label}) Triple Time")

	ax.axvline(1, color='red', linestyle='--', label='1:1 Match')
	ax.set_xlabel(f"Rebound / Starforge ({label})")
	ax.legend()
	ax.grid(True)
	fig.tight_layout()
	fig.savefig(filename)
	plt.close(fig)

# # ECDF with connected line
def plot_ecdf_overlay(data1, data2, label, filename):
	x1 = np.sort(data1)
	y1 = np.arange(1, len(x1)+1) / len(x1)

	x2 = np.sort(data2)
	y2 = np.arange(1, len(x2)+1) / len(x2)

	fig, ax = plt.subplots(figsize=(6, 4))
	ax.plot(x1, y1, label="Original Time", linestyle='-', color='blue')
	ax.plot(x2, y2, label="Triple Time", linestyle='--', color='darkorange')
	ax.axvline(1, color='red', linestyle='--', label='1:1 Match')
	ax.set_title(f"ECDF Comparison ({label}) Triple Time")
	ax.set_xlabel(f"Rebound / Starforge ({label})")
	ax.set_ylabel("Cumulative Probability")
	ax.set_xlim(0,6)
	ax.legend()
	ax.grid(True)
	fig.tight_layout()
	fig.savefig(filename)
	plt.close(fig)


# Save all 6 plots
make_overlay_plot(ratio_ao, ratio_att, pdf_ao, pdf_at, x_vals_a, "Semimajor Axis", "unnormalized", "triple_unnorm_sma.pdf")
make_overlay_plot(ratio_ao, ratio_att, pdf_ao, pdf_at, x_vals_a, "Semimajor Axis", "pdf", "triple_pdf_sma.pdf", xlim=(-3,10))
make_overlay_plot(ratio_eo, ratio_ett, pdf_eo, pdf_et, x_vals_e, "Eccentricity", "unnormalized", "triple_unnorm_ecc.pdf", xlim=(-3,5))
make_overlay_plot(ratio_eo, ratio_ett, pdf_eo, pdf_et, x_vals_e, "Eccentricity", "pdf", "triple_pdf_ecc.pdf", xlim=(-3,5))
plot_ecdf_overlay(ratio_ao, ratio_att, "Semimajor Axis", "triple_overlay_ecdf_sma.pdf")
plot_ecdf_overlay(ratio_eo, ratio_ett, "Eccentricity", "triple_overlay_ecdf_ecc.pdf")
