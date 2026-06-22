
## VERSION WITH ALL ANSWERS 
import torch
from datasets import load_dataset,concatenate_datasets
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator

# 1. Chargement des datasets d'entraînement et de validation
raw_train_dataset_full = load_dataset("json", data_files="datasets/BioASQ_processed2_for_biobert.json")['train']

#preparation du validation set en combinant 1 batch de test et une partie du training set (pour avoir plus d'exemples de validation)
raw_val_dataset = load_dataset("json", data_files="datasets/13B1_processed2.json")['train']

print(f"Number of questions in the validation set: {len(raw_val_dataset)}")

split_dataset = raw_train_dataset_full.train_test_split(test_size=0.1, seed=42)

raw_train_dataset = split_dataset["train"]
raw_val_dataset = concatenate_datasets([split_dataset["test"], raw_val_dataset])

print(f"New Train set length : {len(raw_train_dataset)}")
print(f"New validation length : {len(raw_val_dataset)}")

# Chargement du modèle et du tokenizer
model_name = "dmis-lab/biobert-base-cased-v1.1-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Fonction de tokenisation adaptee aux reponses multiples
def preprocess_training_examples(examples):
    questions = []
    contexts = []
    start_positions = []
    end_positions = []
    
    ####TOKENISATION ##########
    # On cree un exemple d'entraînement pour chaque réponse valide (même question et contexte dupliqués)
    # On boucle sur chaque exemple du batch
    for i in range(len(examples["question"])):
        question = examples["question"][i].strip()
        context = examples["context"][i]
        answer_texts = examples["answers"][i]["text"]
        answer_starts = examples["answers"][i]["answer_start"]

        # On crée un exemple d'entraînement par réponse valide
        for text, start_char in zip(answer_texts, answer_starts):
            questions.append(question)
            contexts.append(context)
            
            # Stockage temporaire des positions de caractères
            start_positions.append(start_char)
            end_positions.append(start_char + len(text))

    # Tokenisation globale des inputs dupliqués (Question + Contexte)
    inputs = tokenizer(
        questions,
        contexts,
        max_length=384,
        truncation="only_second",  # Tronque le contexte si ça dépasse
        padding="max_length",
        return_offsets_mapping=True,
    )

    #### Alignement des positions de caractères aux positions de tokens ####
    offset_mapping = inputs.pop("offset_mapping")
    final_start_positions = []
    final_end_positions = []

    # Calcul des positions au niveau des TOKENS
    for i, offset in enumerate(offset_mapping):
        start_char = start_positions[i]
        end_char = end_positions[i]
        sequence_ids = inputs.sequence_ids(i)

        # Trouver le début et la fin du contexte dans les tokens
        idx = 0
        while idx < len(sequence_ids) and sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # Si la réponse est tronquée ou hors de la fenêtre
        if context_start >= len(sequence_ids) or offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            final_start_positions.append(0)  # CLS token
            final_end_positions.append(0)
        else:
            # Index du token de début
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            final_start_positions.append(idx - 1)

            # Index du token de fin
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            final_end_positions.append(idx + 1)

    inputs["start_positions"] = final_start_positions
    inputs["end_positions"] = final_end_positions
    return inputs

# 3. Application de la tokenisation (avec augmentation de données)
# batched=True est gardé, mais comme le nombre de lignes en sortie change, 
# la suppression des colonnes d'origine evite les conflits de taille.
tokenized_train = raw_train_dataset.map(
    preprocess_training_examples,
    batched=True,
    remove_columns=raw_train_dataset.column_names
)

tokenized_val = raw_val_dataset.map(
    preprocess_training_examples,
    batched=True,
    remove_columns=raw_val_dataset.column_names
)

print(f"Train size after data augmentation: {len(tokenized_train)}")
print(f"Validation size after data augmentation: {len(tokenized_val)}")

# 4. Paramètres d'entraînement
data_collator = DefaultDataCollator() # prend les questions tokenisées, les regroupe par paquets de 6 (la taille du batch) et les transforme dans le format mathématique exact (==> le modele sait comment lire les données ) 

training_args = TrainingArguments(
    output_dir="./biobert_bioasq13b_fact_list_model",
    evaluation_strategy="epoch",       # Evaluation à la fin de chaque epoch
    save_strategy="epoch",       
    learning_rate=2e-5,
    per_device_train_batch_size=6, 
    per_device_eval_batch_size=6,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,           
    load_best_model_at_end=True, 
)

# 5. Lancement du Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val, 
    data_collator=data_collator,
)

# Lancer l'entraînement
trainer.train()


# =========================
# 8. SAVE MODEL
# =========================

model.save_pretrained("./biobert_bioasq13b_fact_list_model")
tokenizer.save_pretrained("./biobert_bioasq13b_fact_list_model")

print("Training finished!")