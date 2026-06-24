import os
import numpy as np
import torch
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ==========================================
# 1. LOAD TEST DATASET (PQA-L SPLIT)
# ==========================================
print("--- Loading Test Dataset (Expert Labeled) ---")
pqa_l = load_dataset("pubmed_qa", "pqa_labeled", split="train")

#classlabel needed for stratifified split 
class_label_feature = ClassLabel(names=["yes", "no", "maybe"])
pqa_l = pqa_l.cast_column("final_decision", class_label_feature)

# Get the same split as in the training script to ensure consistency
pqa_l_split = pqa_l.train_test_split(test_size=0.5, seed=42, stratify_by_column="final_decision") 
raw_test_set = pqa_l_split["test"]


# ==========================================
# 2. LOAD FINAL MODEL FROM PHASE 2
# ==========================================
model_path = "./model_pqa_multiphase"  
print(f"--- Loading model from : {model_path} ---")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


# ==========================================
# 3. PREPARATION OF TEST DATA
# ==========================================
def preprocess_test(examples):
    contexts = [" ".join(c["contexts"]) for c in examples["context"]]
    tokenized = tokenizer(examples["question"], contexts, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = examples["final_decision"]
    return tokenized

print("--- Tokenization of the Test Set ---")
test_dataset = raw_test_set.map(preprocess_test, batched=True)


# ==========================================
# 4. CONFIGURATION OF METRICS FOR EVALUATION
# ==========================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    # Calculation of macro metrics suitable for the strong imbalance
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall
    }


# ==========================================
# 5. INFERENCE VIA TRAINER
# ==========================================
eval_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=16,   
    do_train=False,
    do_eval=True,
    report_to="none"                 
)

trainer = Trainer(
    model=model,
    args=eval_args,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("--- Launching Inference on the Test Set ---")
eval_output = trainer.predict(test_dataset) 

# Extraction of useful data
metrics = eval_output.metrics
logits = eval_output.predictions
labels = eval_output.label_ids
preds = np.argmax(logits, axis=1)


# ==========================================
# 6. DISPLAY AND DETAILED RESULTS + GRAPHIC
# ==========================================
print("\n" + "="*55)
print("             TEST SET GLOBAL METRICS             ")
print("="*55)
print(f"Test Loss:        {metrics.get('test_loss'):.4f}")
print(f"Accuracy:         {metrics.get('test_accuracy') * 100:.2f}%")
print(f"Macro F1-Score:   {metrics.get('test_f1_macro') * 100:.2f}%")
print(f"Macro Precision:  {metrics.get('test_precision_macro') * 100:.2f}%")
print(f"Macro Recall:     {metrics.get('test_recall_macro') * 100:.2f}%")

print("\n" + "="*55)
print("          PER-CLASS DETAILED REPORT              ")
print("="*55)
print(classification_report(labels, preds, target_names=["YES", "NO", "MAYBE"]))

print("="*55)
print("               CONFUSION MATRIX                  ")
print("="*55)
cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
print("True \\ Pred |    YES    |    NO     |   MAYBE   |")
print(f"YES         |   {cm[0][0]:<7} |   {cm[0][1]:<7} |   {cm[0][2]:<7} |")
print(f"NO          |   {cm[1][0]:<7} |   {cm[1][1]:<7} |   {cm[1][2]:<7} |")
print(f"MAYBE       |   {cm[2][0]:<7} |   {cm[2][1]:<7} |   {cm[2][2]:<7} |")
print("="*55)

# ------------------------------------------
# GENERATE SEABORN HEATMAP
# ------------------------------------------
print("\n--- Generation confusion matrix ---")
class_names = ["YES", "NO", "MAYBE"]

plt.figure(figsize=(8, 6))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, 
            yticklabels=class_names,
            cbar=True,
            annot_kws={"size": 12}) 

plt.title("Confusion Matrix - Multiphase finetuning", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Predicted Labels", fontsize=12, labelpad=10)
plt.ylabel("True Labels", fontsize=12, labelpad=10)

plt.tight_layout()

# Save the confusion matrix 
output_image_path = "confusion_matrix_multiphase.png"
plt.savefig(output_image_path, dpi=300)
print(f"Confusion matrix saved on : '{output_image_path}'")

# ------------------------------------------
# GENERATE CLASSIFICATION REPORT HEATMAP
# ------------------------------------------
print("\n--- Generation of Classification Report Heatmap ---")

# 1. Generate the report as a Python dictionary
report_dict = classification_report(labels, preds, target_names=class_names, output_dict=True)

# 2. Convert to Pandas DataFrame and clean for the graphic
#Deleting the 'support' column which has values too large compared to the scores (0 to 1)
df_report = pd.DataFrame(report_dict).iloc[:-1, :] 

# Ignore the 'accuracy', 'macro avg', and 'weighted avg' rows for the heatmap, focusing only on per-class metrics
df_report_classes = df_report.drop(columns=["accuracy", "macro avg", "weighted avg"], errors="ignore")

# 3. Create the heatmap using Seaborn
plt.figure(figsize=(10, 5))

sns.heatmap(df_report_classes.T, 
            annot=True, 
            fmt=".3f", 
            cmap="RdYlGn", 
            vmin=0.0, 
            vmax=1.0,      
            cbar=True,
            annot_kws={"size": 12})

plt.title("Classification Report - Per Class Performance", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Metrics", fontsize=12, labelpad=10)
plt.ylabel("Classes", fontsize=12, labelpad=10)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

plt.tight_layout()

# Save the image
output_report_path = "classification_report_heatmap.png"
plt.savefig(output_report_path, dpi=300)
print(f"Classification report heatmap saved successfully under: '{output_report_path}'")