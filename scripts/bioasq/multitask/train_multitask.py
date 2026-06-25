import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer

# ==================================================
# 1. ARCHITECTURE OF MULTITASK BIOBERT
# ==================================================
class MultitaskBioBERT(nn.Module):
    def __init__(self, model_name_or_path="dmis-lab/biobert-base-cased-v1.1-squad", num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = self.bert.config.hidden_size
        
        self.qa_outputs = nn.Linear(hidden_size, 2) # Span extraction (start and end logits)
        self.classifier = nn.Linear(hidden_size, num_labels) # Classification Yes/No
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask, token_type_ids=None, task_ids=None, start_positions=None, end_positions=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        
        sequence_output = self.dropout(outputs[0]) 
        pooled_output = self.dropout(outputs[1]) 
        
        # Outputs of the QA head (span extraction) and classification head (Yes/No)
        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        classification_logits = self.classifier(pooled_output)
        
        loss = torch.tensor(0.0, device=input_ids.device)
        
        if task_ids is not None:
            loss_fct_qa = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
            loss_fct_cls = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
            
            qa_mask = (task_ids == 0).float()
            cls_mask = (task_ids == 1).float()
            
            # 1. Loss for QA (span extraction) task 
            if start_positions is not None and end_positions is not None:
                start_loss = loss_fct_qa(start_logits, start_positions)
                end_loss = loss_fct_qa(end_logits, end_positions)
                qa_loss = (start_loss + end_loss) / 2
                masked_qa_loss = qa_loss * qa_mask
                if qa_mask.sum() > 0:
                    loss += masked_qa_loss.sum() / qa_mask.sum()
            
            # 2. Loss for classification (Yes/No) task
            if labels is not None:
                cls_loss = loss_fct_cls(classification_logits, labels)
                masked_cls_loss = cls_loss * cls_mask
                if cls_mask.sum() > 0:
                    loss += masked_cls_loss.sum() / cls_mask.sum()

        return {
            "loss": loss, 
            "start_logits": start_logits, 
            "end_logits": end_logits, 
            "classification_logits": classification_logits
        }

# ===================================================
# 2. MAPPING & DATA AUGMENTATION (TOKENISATION)
# ===================================================
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1-squad")

def preprocess_multitask_examples(examples):
    questions, contexts, task_ids, labels = [], [], [], []
    start_positions, end_positions = [], []
    
    for i in range(len(examples["question"])):
        q_type_id = examples["task_id"][i] # 0 for QA, 1 for Yes/No classification
        
        if q_type_id == 1:  # Question Yes/No
            questions.append(examples["question"][i])
            contexts.append(examples["context"][i])
            task_ids.append(1)
            labels.append(examples["label"][i])
            start_positions.append(-1) 
            end_positions.append(-1)
        else:               # Question Factoid/List
            answer_texts = examples["answers"][i]["text"]
            answer_starts = examples["answers"][i]["answer_start"]
            
            # Augmentation intra-snippet : Duplication pour chaque réponse valide
            for text, start_char in zip(answer_texts, answer_starts):
                questions.append(examples["question"][i])
                contexts.append(examples["context"][i])
                task_ids.append(0)
                labels.append(-1) 
                start_positions.append(start_char)
                end_positions.append(start_char + len(text))
                
    inputs = tokenizer(questions, contexts, max_length=384, truncation="only_second", padding="max_length", return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")
    
    # Finding the token positions corresponding to the character-level start 
    # and end positions of the answers
    final_starts, final_ends = [], []
    for idx, offset in enumerate(offset_mapping):
        if task_ids[idx] == 1:
            final_starts.append(-1)
            final_ends.append(-1)
            continue
            
        start_char, end_char = start_positions[idx], end_positions[idx]
        seq_ids = inputs.sequence_ids(idx)
        
        c_start = next((i for i, val in enumerate(seq_ids) if val == 1), len(seq_ids))
        c_end = next((i for i in range(len(seq_ids)-1, -1, -1) if seq_ids[i] == 1), 0)
        
        if c_start >= len(seq_ids) or offset[c_start][0] > start_char or offset[c_end][1] < end_char:
            final_starts.append(-1)
            final_ends.append(-1)
        else:
            t_start = c_start
            while t_start <= c_end and offset[t_start][0] <= start_char: t_start += 1
            final_starts.append(t_start - 1)
            
            t_end = c_end
            while t_end >= c_start and offset[t_end][1] >= end_char: t_end -= 1
            final_ends.append(t_end + 1)
            
    inputs["task_ids"] = task_ids
    inputs["labels"] = labels
    inputs["start_positions"] = final_starts
    inputs["end_positions"] = final_ends
    return inputs

class MultitaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

# =====================================================================
# 3. TRAINING MULTITASK BIOBERT ON BIOASQ DATASET
# =====================================================================
if __name__ == "__main__":
    print("\n--- CHARGEMENT DES FICHIERS JSON PRÉPARÉS ---")
    raw_train = load_dataset("json", data_files="datasets/multitask_bioasq_train.json")['train']
    raw_val = load_dataset("json", data_files="datasets/multitask_bioasq_val.json")['train']
    
    columns_to_remove = raw_train.column_names

    print("Mapping et alignement des tokens (Train)...")
    train_ds = raw_train.map(preprocess_multitask_examples, batched=True, remove_columns=columns_to_remove)
    print("Mapping et alignement des tokens (Validation)...")
    val_ds = raw_val.map(preprocess_multitask_examples, batched=True, remove_columns=columns_to_remove)
    
    print(f"Taille après Tokenisation & Augmentation - Train: {len(train_ds)} | Val: {len(val_ds)}")

    print("\n--- ENTRAÎNEMENT DU MODÈLE MULTITÂCHE ---")
    model = MultitaskBioBERT()
    
    training_args = TrainingArguments(
        output_dir="./multitask_biobert_checkpoint",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=20,
        load_best_model_at_end=True
    )
    
    trainer = MultitaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds
    )
    
    trainer.train()
    
    # Save the final model and tokenizer
    torch.save(model.state_dict(), "./multitask_biobert_model.pt")
    tokenizer.save_pretrained("./multitask_biobert_tokenizer")
    print("Multitask tokenizer saved at `./multitask_biobert_tokenizer`.")
