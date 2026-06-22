import json
import numpy as np
import pandas as pd
from datasets import Dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def load_yesno_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    records = []
    for q in data['questions']:
        if q['type'] != 'yesno':
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
train_dataset = load_yesno_data('datasets/BioASQ-training13b/BioASQ-training13b/training13b.json')
val_dataset = load_yesno_data('datasets/Task13BGoldenEnriched/Task13BGoldenEnriched/13B1_golden.json')

num_train_samples = len(train_dataset)
num_val_samples = len(val_dataset)

model_name = "dmis-lab/biobert-base-cased-v1.1-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

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

clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return clf_metrics.compute(predictions=predictions, references=labels)

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
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

print("Starting model fine-tuning...")
trainer.train()

print("Running final validation evaluation...")
eval_results = trainer.evaluate()

# Save performance configuration to CSV
performance_data = {
    "Stage": ["Validation (13B1)"],
    "Model Source": [model_name],
    "Dataset Samples": [num_val_samples],
    "Loss": [eval_results.get("eval_loss")],
    "Accuracy": [eval_results.get("eval_accuracy")],
    "F1-Score": [eval_results.get("eval_f1")],
    "Precision": [eval_results.get("eval_precision")],
    "Recall": [eval_results.get("eval_recall")]
}

df_performance = pd.DataFrame(performance_data)
df_performance.to_csv("biobert_bioasq13b_yesno_train_performance.csv", index=False)

# Save standalone artifacts
model.save_pretrained("./biobert_bioasq13b_yesno_model")
tokenizer.save_pretrained("./biobert_bioasq13b_yesno_model")
print("Training script finished successfully. Weights and CSV saved.")