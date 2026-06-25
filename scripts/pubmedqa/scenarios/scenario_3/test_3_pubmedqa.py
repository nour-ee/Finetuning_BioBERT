import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ==========================================
# 1. DATA LOADING (PQA-L TEST SPLIT)
# ==========================================
print("--- Loading : Expert Labeled Data (PQA-L) ---")
pqa_l = load_dataset("pubmed_qa", "pqa_labeled", split="train")

# Get the exact split from the training script (seed=42)
pqa_l_split = pqa_l.train_test_split(test_size=0.5, seed=42) 
raw_test_set = pqa_l_split["test"]

# ==========================================
# 2. LOAD TRAINED MODEL & TOKENIZER
# ==========================================
model_path = "./pqa_a_finetuned_model"  # Path to the model fine-tuned on only PQA-A in the training script

print(f"--- Loading trained model and tokenizer from: {model_path} ---")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# ==========================================
# 3. PREPROCESSING TEST SET 
# ==========================================
label_map = {"yes": 0, "no": 1, "maybe": 2}

def preprocess_test(examples):
    # Same preprocessing function as used during training to ensure consistency
    contexts = [" ".join(c["contexts"]) for c in examples["context"]]
    tokenized = tokenizer(examples["question"], contexts, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = [label_map[l] for l in examples["final_decision"]]
    return tokenized

print("--- Preprocessing test set ---")
test_dataset = raw_test_set.map(preprocess_test, batched=True)

# ==========================================
# 4. COMPUTE METRICS FOR IMBALANCED DATA
# ==========================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall
    }

# ==========================================
# 5. INITIALIZE STANDARD TRAINER & EVALUATE
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

print("--- Running Evaluation on Test Set ---")
metrics = trainer.evaluate()

# ==========================================
# 6. DISPLAY RESULTS
# ==========================================
print("\n" + "="*40)
print("       TEST SET EVALUATION RESULTS       ")
print("="*40)
print(f"Test Loss:        {metrics.get('eval_loss'):.4f}")
print(f"Accuracy:         {metrics.get('eval_accuracy') * 100:.2f}%")
print(f"Macro F1-Score:   {metrics.get('eval_f1_macro') * 100:.2f}%")
print(f"Macro Precision:  {metrics.get('eval_precision_macro') * 100:.2f}%")
print(f"Macro Recall:     {metrics.get('eval_recall_macro') * 100:.2f}%")
print("="*40)