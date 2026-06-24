import os
import shutil
from collections import Counter
import numpy as np
import torch
from torch import nn
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# ==========================================
#  INITIALIZATION & CONFIGURATION
# ==========================================
model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(examples):
    # Concat contexts into a single string for each example
    contexts = [" ".join(c["contexts"]) for c in examples["context"]]
    tokenized = tokenizer(examples["question"], contexts, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = examples["final_decision"]
    return tokenized

# ==========================================
#  WEIGHTED LOSS TRAINER CLASS
# ==========================================
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Transformation des poids en tenseur PyTorch
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            # Apply class weights to the loss function
            weights = self.class_weights.to(labels.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ==========================================
# 3. LOAD & PREPARE DATASETS
# ==========================================
print("--- Loading datasets from Hugging Face ---")
pqa_a = load_dataset("pubmed_qa", "pqa_artificial", split="train")
pqa_l = load_dataset("pubmed_qa", "pqa_labeled", split="train")

# Cast to ClassLabel to ensure labels are indexed as [0, 1, 2] (yes, no, maybe)
class_label_feature = ClassLabel(names=["yes", "no", "maybe"])
pqa_a = pqa_a.cast_column("final_decision", class_label_feature)
pqa_l = pqa_l.cast_column("final_decision", class_label_feature)

# Stratified split ensures that the distribution of labels is preserved in both subsets
pqa_l_split = pqa_l.train_test_split(test_size=0.5, seed=42, stratify_by_column="final_decision") 
# Sub-split to isolate the validation set (10% of the training subset)
train_val_split = pqa_l_split["train"].train_test_split(test_size=0.1, seed=42, stratify_by_column="final_decision")

raw_pqa_l_train = train_val_split["train"]
raw_val_set = train_val_split["test"]

print(f"Size PQA-A : {len(pqa_a)} | PQA-L Train : {len(raw_pqa_l_train)} | PQA-L Val : {len(raw_val_set)}")

# Tokenization
print("--- Tokenization of subsets  ---")
tokenized_pqa_a = pqa_a.map(preprocess, batched=True)
tokenized_pqa_l_train = raw_pqa_l_train.map(preprocess, batched=True)
val_dataset = raw_val_set.map(preprocess, batched=True)


# ==========================================
# 4. PHASE 1 : FINE-TUNING ON PQA-A
# ==========================================
print("\n" + "="*60)
print(">>> PHASE 1 : Fine-tuning on PQA-A  <<<")
print("="*60)

# Load BioBERT model 
model_phase1 = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Training arguments for Phase 1 
args_phase1 = TrainingArguments(
    output_dir="./biobert_pqa_phase1_output",
    per_device_train_batch_size=16,
    num_train_epochs=1,          
    learning_rate=3e-5,
    logging_steps=500,
    save_strategy="no",          
    report_to="none"
)

# Launch training for Phase 1
trainer_phase1 = Trainer(
    model=model_phase1,
    args=args_phase1,
    train_dataset=tokenized_pqa_a
)

trainer_phase1.train()

# Save the model from Phase 1 to a temporary directory for later use in Phase 2
tmp_phase1_dir = "./tmp_model_phase1"
print(f"--- Save Phase 1 on : {tmp_phase1_dir} ---")
trainer_phase1.save_model(tmp_phase1_dir)

# Free up GPU memory 
del model_phase1
del trainer_phase1
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("--- End Phase 1 ---")


# ==========================================
# 5. PHASE 2 : FINE-TUNING DE SPECIALISATION (PQA-L)
# ==========================================
print("\n" + "="*60)
print(">>> PHASE 2 : Fine-tuning on  PQA-L  <<<")
print("="*60)

# Load the model from Phase 1 to continue fine-tuning on the PQA-L dataset
print(f"--- Loading the adapted model from Phase 1 ({tmp_phase1_dir}) ---")
model_phase2 = AutoModelForSequenceClassification.from_pretrained(tmp_phase1_dir)

# Calculation of class weights based on the distribution in the PQA-L training set
pqa_l_counts = Counter(raw_pqa_l_train["final_decision"])
total_pqa_l = len(raw_pqa_l_train)

weights = [
    total_pqa_l / (3 * max(pqa_l_counts.get(k, 1), 1))
    for k in [0, 1, 2]
]
# Normalisation to set the majority class ('yes') weight to 1.0
weights = [w / weights[0] for w in weights]

distribution_report = (
    "--------------------------------------------------\n"
    f"Distribution Phase 2 (PQA-L Train - {total_pqa_l} examples):\n"
    f"  YES: {pqa_l_counts[0]} | NO: {pqa_l_counts[1]} | MAYBE: {pqa_l_counts[2]}\n"
    f"Weight for the Loss (YES, NO, MAYBE): {[round(w, 2) for w in weights]}\n"
    "--------------------------------------------------\n"
)
print(distribution_report)

# TrainingArguments for Phase 2 with class weights to address imbalance
args_phase2 = TrainingArguments(
    output_dir="./biobert_pqa_phase2_output",
    per_device_train_batch_size=8,   
    per_device_eval_batch_size=16,
    num_train_epochs=6,              
    learning_rate=2e-5,              
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,     
    metric_for_best_model="loss",
    report_to="none"
)


trainer_phase2 = WeightedLossTrainer(
    model=model_phase2, 
    args=args_phase2,
    train_dataset=tokenized_pqa_l_train,
    eval_dataset=val_dataset,
    class_weights=weights
)

trainer_phase2.train()


# ==========================================
# 6. SAVE FINAL MODEL & CLEANUP
# ==========================================
output_model_dir = "./model_pqa_multiphase"
print(f"\n--- Saving final model in : {output_model_dir} ---")
trainer_phase2.save_model(output_model_dir)
tokenizer.save_pretrained(output_model_dir)

# Delete temporary Phase 1 directory to free up space
if os.path.exists(tmp_phase1_dir):
    shutil.rmtree(tmp_phase1_dir)

# Report text summary of the training phases and class distribution
with open("training_summary.txt", "w", encoding="utf-8") as f:
    f.write("=== MULTIPHASE :  FINE-TUNING SUMMARY ===\n")
    f.write(distribution_report)

print("Global process completed. Your final model is ready for the testing script!")