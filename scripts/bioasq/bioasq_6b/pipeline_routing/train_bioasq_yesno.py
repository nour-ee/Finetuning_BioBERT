import json
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# ========================
#  LOAD DATASET 
# ========================

def load_yesno_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = []
    for q in data['questions']:
        if q['type'] != 'yesno': # select only yes/no questions
            continue
            
        snippets = [s['text'].strip() for s in q.get('snippets', [])]
        context = " ".join(snippets).strip()
        
        if not context:
            continue
            
        raw_answer = str(q.get('exact_answer', '')).lower()
        label = 1 if 'yes' in raw_answer else 0
        
        records.append({
            "id": q['id'],
            "context": context,
            "question": q['body'],
            "label": label
        })
        
    return Dataset.from_pandas(pd.DataFrame(records))

print("Loading training and validation datasets...")
train_dataset = load_yesno_data('datasets/BioASQ-training6b/BioASQ-trainingDataset6b.json')
val_dataset = load_yesno_data('datasets/Task6BGoldenEnriched/6B1_golden.json')

num_train_samples = len(train_dataset)
num_val_samples = len(val_dataset)

# =========================
#  LOAD MODEL AND TOKENIZER
# =========================

model_name = "dmis-lab/biobert-base-cased-v1.1-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Apply tokenization to the datasets
def tokenize_yesno(examples):
    return tokenizer(
        examples["question"],
        examples["context"],
        max_length=384,
        truncation="only_second",
        padding="max_length"
    )

tokenized_train = train_dataset.map(tokenize_yesno, batched=True)
tokenized_val = val_dataset.map(tokenize_yesno, batched=True)

training_args = TrainingArguments(
    output_dir="./biobert_bioasq13b_yesno_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val
)

print("Starting model fine-tuning...")
trainer.train()

# ==============================
# SAVE MODEL AND TOKENIZER
#===============================
model.save_pretrained("./biobert_bioasq6b_yesno_model")
tokenizer.save_pretrained("./biobert_bioasq6b_yesno_model")
print("Training script finished successfully.")
