import starfile
import pandas as pd
import matplotlib.pyplot as plt

# === Load STAR file ===
star_path = "H1_DoubleLinker_annotated.star"
df = starfile.read(star_path)  # DataFrame

# === Count number of particles per component ===
comp_sizes = df['rlnLC_ChainComponent'].value_counts().sort_index()
# Filter: only keep components with size >= 2
comp_sizes = comp_sizes[comp_sizes >= 2]


# === Save raw component distribution (component_id + size) ===
comp_sizes.to_csv("chain_component_distribution.csv", header=["count"])
print("CSV saved to chain_component_distribution.csv")


# === Distribution of component sizes (histogram of counts of sizes) ===
size_counts = comp_sizes.value_counts().sort_index()

# Calculate average and max size
avg_size = comp_sizes.mean()
max_size = comp_sizes.max()

# === Plot ===
plt.figure(figsize=(4,4))
plt.bar(size_counts.index, size_counts.values, color="#e0f3f8", edgecolor="black") # lightgray, skyblue, #e0f3f8

plt.yscale("log")
plt.xlabel("Number of connected nucleosomes", fontsize=13)
plt.ylabel("Number of occurrences", fontsize=13)
# plt.title("Tomogram 1", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


# Annotate average and max size
plt.text(0.95, 0.95, f"Average size: {avg_size:.2f}\nMax size: {max_size}",
         transform=plt.gca().transAxes,
         ha="right", va="top", fontsize=12)

plt.tight_layout()

# Save figure
out_file = "chain_component_size_distribution.png"
plt.savefig(out_file, dpi=300)
print(f"Figure saved to {out_file}")

plt.show()
