import pandas as pd
import sys
import os

def split_csv(input_csv):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)

    if "tomo_name" not in df.columns:
        print("Error: 'tomo_name' column not found in the CSV.")
        return

    tomo_names = df["tomo_name"].unique()
    print(f"Found {len(tomo_names)} unique tomograms: {', '.join(map(str, tomo_names))}")

    # Create a subfolder to keep the workspace clean
    out_dir = "split_by_tomo"
    os.makedirs(out_dir, exist_ok=True)

    for tomo in tomo_names:
        sub_df = df[df["tomo_name"] == tomo]
        # Clean the tomo name just in case it has weird characters
        safe_tomo_name = str(tomo).replace("/", "_").replace("\\", "_")
        out_name = os.path.join(out_dir, f"{safe_tomo_name}_edges.csv")
        sub_df.to_csv(out_name, index=False)
        print(f"Saved {len(sub_df)} edges to {out_name}")

    print(f"\nAll split files have been saved in the '{out_dir}' directory.")

if __name__ == "__main__":
    # Default to DoubleLinker_edges.csv if no argument is provided
    csv_file = "Linker_edges.csv"
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    split_csv(csv_file)
