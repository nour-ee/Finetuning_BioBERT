import os
from collections import Counter
import numpy as np
import torch
from torch import nn
from datasets import load_dataset, concatenate_datasets,ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(examples):
    # Concatenate the medical contexts into a single string for each example
    contexts = [" ".join(c["contexts"]) for c in examples["context"]]
    tokenized = tokenizer(examples["question"], contexts, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = examples["final_decision"]
    return tokenized

# ==========================================
# 2. CUSTOM TRAINER FOR DYNAMIC CLASS WEIGHTS
# ==========================================
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Save the dynamically calculated class weights for use in the loss function
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            weights = self.class_weights.to(labels.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
        # calculating the loss using the logits and labels
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1)) 
        return (loss, outputs) if return_outputs else loss


# ==========================================
# 3. DATA LOADING & SPLITTING
# ==========================================
print("--- Loading : Artificial Data (PQA-A) ---")
pqa_a = load_dataset("pubmed_qa", "pqa_artificial", split="train")

print("--- Loading : Expert Labeled Data (PQA-L) ---")
pqa_l = load_dataset("pubmed_qa", "pqa_labeled", split="train")

#Required for stratified splitting --> cast the final_decision column to ClassLabel for both datasets
class_label_feature = ClassLabel(names=["yes", "no", "maybe"])
# Cast the final_decision column to ClassLabel for both datasets
pqa_l = pqa_l.cast_column("final_decision", class_label_feature)
pqa_a = pqa_a.cast_column("final_decision", class_label_feature)

# Split de PQA-L (50% train/val, 50% test)
pqa_l_split = pqa_l.train_test_split(test_size=0.5, seed=42,stratify_by_column="final_decision") 
# Sub-split for validation (10% of the training subset)
train_val_split = pqa_l_split["train"].train_test_split(test_size=0.1, seed=42, stratify_by_column="final_decision")

raw_pqa_l_train = train_val_split["train"]
raw_val_set = train_val_split["test"]

# CONCATENATION : PQA-A + Train split de PQA-L
print("--- Concatenating PQA-A and Train Split of PQA-L ---")
raw_combined_train = concatenate_datasets([pqa_a, raw_pqa_l_train])
#A SUPPRIMER : Vérification de la distribution des labels dans le jeu de données combiné
# #======================================
# # Print the unique integer codes present in the training column
# print("Unique label integers:", raw_combined_train.unique("final_decision"))
# # Print the  5 first rows of the training dataset to verify the labels
# print("First 5 rows of the training dataset:", raw_combined_train[:5])
# # Output should look like: [0, 1, 2]

# ==========================================
# 4. DISTRIBUTION & WEIGHTS CALCULATION
# ==========================================
pqa_a_counts = Counter(pqa_a["final_decision"])
val_counts = Counter(raw_val_set["final_decision"])
combined_train_counts = Counter(raw_combined_train["final_decision"])

# Calculate weights to handle imbalance
total_samples = len(raw_combined_train)
num_classes = 3
weights = [
    total_samples / (num_classes * max(combined_train_counts.get(k, 1), 1))
    for k in [0, 1, 2]
]
# Normalization to ensure the weight of the majority class ('yes') is equal to 1.0
weights = [w / weights[0] for w in weights]

distribution_report = (
    "--------------------------------------------------\n"
    f"Combined Train Set (PQA-A + PQA-L Train) Size: {total_samples} samples:\n"
    f"  YES: {combined_train_counts[0]} | NO: {combined_train_counts[1]} | MAYBE: {combined_train_counts[2]}\n"
    f"Calculated Class Weights (YES, NO, MAYBE): {[round(w, 2) for w in weights]}\n"
    f"Validation Set (PQA-L Validation) Size: {len(raw_val_set)} samples:\n"
    f"  YES: {val_counts[0]} | NO: {val_counts[1]} | MAYBE: {val_counts[2]}\n"
    "--------------------------------------------------\n"
)
print(distribution_report)

# ==========================================
# 5. PREPROCESSING (TOKENIZATION)
# ==========================================
print("--- Tokenizing Datasets ---")
train_dataset = raw_combined_train.map(preprocess, batched=True)
val_dataset = raw_val_set.map(preprocess, batched=True)

# ==========================================
# 6. METADATA WRITING
# ==========================================
summary_file = "training_pubmedqa_scenario2_summary.txt"
with open(summary_file, "w", encoding="utf-8") as f:
    f.write("=== PUBMEDQA SCENARIO 2 TRAINING SUMMARY ===\n")
    f.write(f"Base Model: {model_name}\n\n")
    f.write(distribution_report)

# ==========================================
# 7. MODEL & TRAINING ARGUMENTS
# ==========================================
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

training_args = TrainingArguments(
    output_dir="./biobert_pqa_scenario2",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=3e-5,
    logging_steps=50,
    evaluation_strategy="epoch",        
    save_strategy="epoch",        
    load_best_model_at_end=True   
)

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    class_weights=weights
)


print("--- Starting Training ---")
trainer.train()

# ==========================================
# 8. SAVE FINAL MODEL
# ==========================================
output_model_dir = "./model_pqa_scenario2"
print(f"--- Save final model and tokenizer in {output_model_dir} ---")

trainer.save_model(output_model_dir)
tokenizer.save_pretrained(output_model_dir)

print("Training and saving completed successfully!")
