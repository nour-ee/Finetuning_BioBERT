import os
import csv
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# 1. Load the SAVED finalized model and tokenizer
model_path = "./pqa_a_finetuned_model"
print(f"--- Loading Saved Model from {model_path} ---")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

label_map = {"yes": 0, "no": 1, "maybe": 2}

def preprocess_fn(examples):
    contexts = [" ".join(c["contexts"]) for c in examples["context"]]
    tokenized = tokenizer(examples["question"], contexts, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = [label_map[l] for l in examples["final_decision"]]
    return tokenized

# ==========================================
# ISOLATE THE TEST DATA
# ==========================================
print("--- Loading Test Dataset ---")
pqa_l_data = load_dataset("pubmed_qa", "pqa_labeled", split="train")

print("--- Tokenizing Test Dataset ---")
tokenized_pqa_l = pqa_l_data.map(preprocess_fn, batched=True)
test_set_size = len(tokenized_pqa_l)

# ==========================================
# ADVANCED METRICS FOR IMBALANCED DATA 
# ==========================================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    
    # Métriques issues du second script (macro-averaged)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    
    # Conservation du rapport détaillé pour la console
    print("\n--- Detailed Classification Report ---")
    print(classification_report(labels, preds, target_names=["yes", "no", "maybe"], zero_division=0))
    
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall
    }

# ==========================================
# EVALUATE
# ==========================================
print("--- Running Final Evaluation ---")
test_args = TrainingArguments(
    output_dir="./pubmedqa_results",
    per_device_eval_batch_size=8,
    logging_steps=10,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=test_args,
    compute_metrics=compute_metrics
)

# Passage du dataset préprocessé
results = trainer.evaluate(tokenized_pqa_l)

# Extraction des nouvelles clés générées par Hugging Face (préfixées par 'eval_')
final_accuracy = results.get("eval_accuracy")
final_f1_macro = results.get("eval_f1_macro")
final_precision_macro = results.get("eval_precision_macro")
final_recall_macro = results.get("eval_recall_macro")
final_loss = results.get("eval_loss")

print("\n" + "="*40)
print("       TEST SET EVALUATION RESULTS       ")
print("="*40)
print(f"RAW TEST LOSS:      {final_loss:.4f}" if final_loss is not None else "RAW TEST LOSS:      N/A")
print(f"ACCURACY:           {final_accuracy * 100:.2f}%" if final_accuracy is not None else "ACCURACY:           N/A")
print(f"MACRO F1-SCORE:     {final_f1_macro * 100:.2f}%" if final_f1_macro is not None else "MACRO F1-SCORE:     N/A")
print(f"MACRO PRECISION:    {final_precision_macro * 100:.2f}%" if final_precision_macro is not None else "MACRO PRECISION:    N/A")
print(f"MACRO RECALL:       {final_recall_macro * 100:.2f}%" if final_recall_macro is not None else "MACRO RECALL:       N/A")
print("="*40)

# ==========================================
# EXPORT METRICS TO CSV
# ==========================================
csv_file_path = "pubmedqa_evaluation_metrics.csv"
print(f"--- Exporting metrics to {csv_file_path} ---")

base_model_architecture = model.config.architectures[0] if model.config.architectures else "BioBERT"

# Mise à jour des entêtes et des données pour inclure Précision et Rappel
csv_headers = [
    "Model Architecture", "Subset Evaluated", "Test Set Size", 
    "Raw Accuracy", "Macro F1-Score", "Macro Precision", "Macro Recall", "Test Loss"
]
csv_data_row = [
    base_model_architecture,
    "pqa_labeled (Unseen Split)",
    test_set_size,
    round(final_accuracy, 4) if final_accuracy is not None else "N/A",
    round(final_f1_macro, 4) if final_f1_macro is not None else "N/A",
    round(final_precision_macro, 4) if final_precision_macro is not None else "N/A",
    round(final_recall_macro, 4) if final_recall_macro is not None else "N/A",
    round(final_loss, 4) if final_loss is not None else "N/A"
]

file_exists = os.path.isfile(csv_file_path)
with open(csv_file_path, mode="a" if file_exists else "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(csv_headers)
    writer.writerow(csv_data_row)

print(f"Metrics successfully written to '{csv_file_path}'!")
