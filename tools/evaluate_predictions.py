import pandas as pd
import argparse
import sys

def extract_edges(csv_path, strict=True):
    """
    Extracts edges from a prediction CSV file.
    
    Args:
        csv_path (str): Path to the DoubleLinker_edges.csv
        strict (bool): 
            If False (Particle-level): Edge is defined as an undirected pair of particle IDs (min_id, max_id).
            If True (Arm-level): Edge is defined as an undirected pair including arm usage: 
                                 ((min_id, arm_for_min), (max_id, arm_for_max)).
                                 
    Returns:
        set: A set of edges.
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return set()
    except FileNotFoundError:
        print(f"Error: File not found ({csv_path})")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        return set()

        
    edges = set()
    for _, row in df.iterrows():
        id_i = row['i_id']
        id_j = row['j_id']
        arm_i = row['arm_i']
        arm_j = row['arm_j']
        
        if not strict:
            # Particle-level edge (undirected)
            edge = tuple(sorted([id_i, id_j]))
        else:
            # Arm-level edge (undirected, including specific ports)
            node_i = (id_i, arm_i)
            node_j = (id_j, arm_j)
            edge = tuple(sorted([node_i, node_j]))
            
        edges.add(edge)
        
    return edges

def evaluate_predictions(truth_path, pred_path, strict=True):
    """
    Compares prediction edges against ground truth edges and calculates metrics.
    """
    truth_edges = extract_edges(truth_path, strict=strict)
    pred_edges = extract_edges(pred_path, strict=strict)
    
    # Calculate True Positives, False Positives, False Negatives
    tp_edges = pred_edges.intersection(truth_edges)
    fp_edges = pred_edges - truth_edges
    fn_edges = truth_edges - pred_edges
    
    tp = len(tp_edges)
    fp = len(fp_edges)
    fn = len(fn_edges)
    
    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("-" * 50)
    print(f"Evaluation Mode : {'Strict (Arm-Level)' if strict else 'Relaxed (Particle-Level)'}")
    print(f"Ground Truth    : {truth_path} ({len(truth_edges)} edges)")
    print(f"Predictions     : {pred_path} ({len(pred_edges)} edges)")
    print("-" * 50)
    print(f"True Positives  (TP) : {tp}")
    print(f"False Positives (FP) : {fp}")
    print(f"False Negatives (FN) : {fn}")
    print("-" * 50)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("-" * 50)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Linker predictions against Ground Truth.")
    parser.add_argument("--truth", required=True, help="Path to the Ground Truth CSV file.")
    parser.add_argument("--pred", required=True, help="Path to the Prediction CSV file.")
    parser.add_argument("--relaxed", action="store_true", help="Use particle-level matching instead of strict arm-level matching.")
    
    args = parser.parse_args()
    
    evaluate_predictions(args.truth, args.pred, strict=not args.relaxed)
