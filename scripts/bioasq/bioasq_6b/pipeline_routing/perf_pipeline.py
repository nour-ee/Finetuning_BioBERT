import pandas as pd
import ast
import re
import os
from sklearn.metrics import f1_score 

def clean_eval_string(text):
    """ Clean the medical text by removing punctuation and extra spaces. """
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Delete visual residues of tokenization (spaces around dashes/commas)
    text = re.sub(r'\s*([-,.\/])\s*', r'\1', text)
    # Delete punctuation to keep only alphanumeric characters and spaces
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

def extract_valid_options(raw_truth):
    """ Cuts and extracts all valid text strings from the ground truth, 
    regardless of the level of bracket nesting. """
    text_raw = str(raw_truth).strip()
    if text_raw == "N/A" or not text_raw:
        return []
        
    # Find all valid text strings enclosed in single or double quotes
    options = re.findall(r"['\"]([^'\"]+)['\"]", text_raw)
    
    # If the regex finds nothing (no quotes), we clean the raw string
    if not options:
        cleaned = text_raw.replace('[', '').replace(']', '').strip()
        if cleaned:
            options = [cleaned]
            
    return [clean_eval_string(opt) for opt in options if opt]

def calculate_pipeline_performance(csv_filepath, output_csv="./results/pipeline_metrics_summary.csv"):
    df = pd.read_csv(csv_filepath)
    
    # We eliminate fallback or error lines
    df = df[~df["Pipeline Predicted Answer"].str.contains("fallback|Skipped|Error", na=False, case=False)]
    
    print(f"Evaluation on {len(df)} valid questions (Yes/No, Factoid, List)...")
    
    metrics_data = {"Metric Name": [], "Question Type": [], "Value": []}
    
    # ==========================================
    # EVALUATION YES/NO
    # ==========================================
    df_yn = df[df["Question Type"] == "yesno"]
    if not df_yn.empty:
        y_true = df_yn["Ground Truth Answer"].apply(lambda x: str(x).lower().strip("[]'\" "))
        y_pred = df_yn["Pipeline Predicted Answer"].str.lower().str.strip()
        
        accuracy_yn = (y_true == y_pred).mean()
        f1_yn = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        print(f"\n--- PERFORMANCE YES/NO (Total: {len(df_yn)}) ---")
        print(f"Accuracy    : {accuracy_yn * 100:.2f}%")
        print(f"F1-Score : {f1_yn * 100:.2f}%")
        
        metrics_data["Metric Name"].extend(["Accuracy", "F1-Score"])
        metrics_data["Question Type"].extend(["yesno", "yesno"])
        metrics_data["Value"].extend([accuracy_yn, f1_yn])
    
    # ==========================================
    # EVALUATION FACTOID
    # ==========================================
    df_fact = df[df["Question Type"] == "factoid"]
    if not df_fact.empty:
        exact_matches = 0
        for _, row in df_fact.iterrows():
            pred = clean_eval_string(row["Pipeline Predicted Answer"])
            if "no extraction found" in pred or not pred:
                continue
                
            # Extract valid options from the ground truth, regardless of bracket nesting
            truths = extract_valid_options(row["Ground Truth Answer"])
            
            # Calculate strict accuracy based on exact matches
            if any(pred in t or t in pred for t in truths if t):
                exact_matches += 1
                
        strict_accuracy = exact_matches / len(df_fact)
        print(f"\n--- PERFORMANCE FACTOID (Total: {len(df_fact)}) ---")
        print(f"Strict Accuracy (Flexible Match) : {strict_accuracy * 100:.2f}%")
        metrics_data["Metric Name"].append("Strict Accuracy (EM)")
        metrics_data["Question Type"].append("factoid")
        metrics_data["Value"].append(strict_accuracy)

    # ==========================================
    # EVALUATION LIST 
    # ==========================================
    df_list = df[df["Question Type"] == "list"]
    if not df_list.empty:
        total_precision = 0
        total_recall = 0
        
        for _, row in df_list.iterrows():
            raw_pred = str(row["Pipeline Predicted Answer"]).strip()
            if not raw_pred or "no extraction found" in raw_pred.lower():
                continue
                
            truths = extract_valid_options(row["Ground Truth Answer"])
            
            # Split the string by the pipe "|" and clean each extracted entity
            preds_list = [clean_eval_string(p) for p in raw_pred.split("|")]
            preds_list = [p for p in preds_list if p]  # Filter out any remaining empty strings
            
            if not preds_list:
                continue
            
            true_positives = 0
            # Calculate true positives by checking if any prediction matches any ground truth answer
            for p in preds_list:
                if any(p in t or t in p for t in truths if t):
                    true_positives += 1
            
            # Calculate precision and recall for the current question
            precision = true_positives / len(preds_list) if len(preds_list) > 0 else 0
            recall = true_positives / max(1, len(truths))
            
            total_precision += precision
            total_recall += recall
            
        mean_precision = total_precision / len(df_list)
        mean_recall = total_recall / len(df_list)
        # Calculate F1-score for the list questions
        f1_list = (2 * mean_precision * mean_recall) / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0
        
        print(f"\n--- PERFORMANCE LIST (Total: {len(df_list)}) ---")
        print(f"Precision avg : {mean_precision * 100:.2f}%")
        print(f"Recall avg      : {mean_recall * 100:.2f}%")
        print(f"F1-Score avg    : {f1_list * 100:.2f}%")
        
        metrics_data["Metric Name"].extend(["Precision", "Recall", "F1-Score"])
        metrics_data["Question Type"].extend(["list", "list", "list"])
        metrics_data["Value"].extend([mean_precision, mean_recall, f1_list])

    # ==========================================
    # SAVE METRICS TO CSV
    # ==========================================
    if metrics_data["Metric Name"]:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics["Value (%)"] = df_metrics["Value"] * 100
        df_metrics = df_metrics.drop(columns=["Value"])
        df_metrics.to_csv(output_csv, index=False)
        print(f"\n[INFO] Metrics saved to : {output_csv}")

if __name__ == "__main__":
    input_file = "./results/pipeline_predictions_combined_6B2_6B5.csv"
    if os.path.exists(input_file):
        calculate_pipeline_performance(input_file)
    else:
        print(f"[ERROR] The predictions file {input_file} does not exist. Please generate it first by running the pipeline script.")