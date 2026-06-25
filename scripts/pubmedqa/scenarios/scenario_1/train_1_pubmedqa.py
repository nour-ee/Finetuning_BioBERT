# finetuning biobert model for question answering on pqa-a dataset 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from torch import nn
from collections import Counter
# 1. Initialize BioBERT Architecture
model_name = "dmis-lab/biobert-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
label_map = {"yes": 0, "no": 1, "maybe": 2}

def preprocess(examples): # Function to tokenize and encode the dataset examples for training
    contexts = [" ".join(c["contexts"]) for c in examples["context"]]
    tokenized = tokenizer(examples["question"], contexts, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = [label_map[l] for l in examples["final_decision"]]
    return tokenized

# ==========================================
# CUSTOM TRAINER FOR CLASS WEIGHTS
# ==========================================
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits") 
        
        # FIX MULTI-GPU: On récupère le device directement depuis le tenseur des labels
        device = labels.device 
        
        # YES is 92.8% of the data, NO is 7.2%, MAYBE is 0% in Phase 1.
        # Imposing ~13x higher penalty when the model gets 'NO' wrong.
        weights = torch.tensor([1.0, 13.0, 13.0], device=device)  
        
        loss_fct = nn.CrossEntropyLoss(weight=weights) 
        
        # FIX MULTI-GPU: Utilisation de self.model.config au lieu de model.config
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1)) 
        return (loss, outputs) if return_outputs else loss
    
# ==========================================
#  Finetuning on Artificial Subset (PQA-A)
# ==========================================
print("--- Loading: Artificial Data ---")
# https://huggingface.co/datasets/qiaojin/PubMedQA 
pqa_a = load_dataset("qiaojin/pubmed_qa", "pqa_artificial", split="train")

# counting the distribution of labels in PQA-A to confirm imbalance
pqa_a_counts = Counter(pqa_a["final_decision"])
print(f"Artificial Dataset Label Distribution: {dict(pqa_a_counts)}")

encoded_pqa_a = pqa_a.map(preprocess, batched=True) # Tokenize and encode the artificial dataset for training

# Initialize the model for sequence classification with 3 labels (YES, NO, MAYBE)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

#FINETUNING WITH THE CUSTOM WEIGHTED LOSS TRAINER TO ADDRESS THE IMBALANCE IN PQA-A
# hyperparameter values : https://huggingface.co/Dinithi/BioBERT 
args_fine_pqa_a = TrainingArguments(
    output_dir="./biobert_pqa_artificial",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    learning_rate=3e-5,
    save_strategy="no"
)

# Use WeightedLossTrainer to handle the PQA-A imbalance
trainer_pqa_a = WeightedLossTrainer(model=model, args=args_fine_pqa_a, train_dataset=encoded_pqa_a)
trainer_pqa_a.train()
trainer_pqa_a.save_model("./pqa_a_finetuned_model") 
tokenizer.save_pretrained("./pqa_a_finetuned_model")
