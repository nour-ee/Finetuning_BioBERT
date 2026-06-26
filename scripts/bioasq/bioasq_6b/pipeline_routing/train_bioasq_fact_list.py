from datasets import load_dataset,concatenate_datasets
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator

# ==========================
#  LOAD DATASET
# =========================

# Load the training dataset and validation dataset
raw_train_dataset_full = load_dataset("json", data_files="datasets/BioASQ6b_processed_for_biobert.json")['train']

# Prepare the validation dataset by combining one batch of test and a part of the training set (to have more validation examples)
raw_val_dataset = load_dataset("json", data_files="datasets/6B1_processed.json")['train']

print(f"Number of questions in the validation set: {len(raw_val_dataset)}")

split_dataset = raw_train_dataset_full.train_test_split(test_size=0.1, seed=42)

raw_train_dataset = split_dataset["train"]
raw_val_dataset = concatenate_datasets([split_dataset["test"], raw_val_dataset])

print(f"New Train set length : {len(raw_train_dataset)}")
print(f"New validation length : {len(raw_val_dataset)}")

# =========================
#  LOAD MODEL AND TOKENIZER
# =========================

# Load the BioBERT model and tokenizer for question answering
model_name = "dmis-lab/biobert-base-cased-v1.1-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# ==========================
#  PREPROCESSING 
# ==========================

# Function to preprocess training examples for question answering with multiple answers (list questions)
def preprocess_training_examples(examples):
    questions = []
    contexts = []
    start_positions = []
    end_positions = []
    
    ####TOKENISATION ##########
    # Create a raw per-answer with duplicated questions and contexts
    for i in range(len(examples["question"])):
        question = examples["question"][i].strip()
        context = examples["context"][i]
        answer_texts = examples["answers"][i]["text"]
        answer_starts = examples["answers"][i]["answer_start"]

        for text, start_char in zip(answer_texts, answer_starts):
            questions.append(question)
            contexts.append(context)
            
            # Storage of character positions 
            start_positions.append(start_char)
            end_positions.append(start_char + len(text))

    # Tokenization of inputs (Question + Context) with truncation and padding
    inputs = tokenizer(
        questions,
        contexts,
        max_length=384,
        truncation="only_second",  
        padding="max_length",
        return_offsets_mapping=True,
    )

    #### Align the character positions to the token positions ####
    offset_mapping = inputs.pop("offset_mapping")
    final_start_positions = []
    final_end_positions = []

    # Calculate the token positions for each answer
    for i, offset in enumerate(offset_mapping):
        start_char = start_positions[i]
        end_char = end_positions[i]
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context in the tokenized input
        idx = 0
        while idx < len(sequence_ids) and sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is truncated or outside the window, set the start and end positions to 0 (CLS token)
        if context_start >= len(sequence_ids) or offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            final_start_positions.append(0)  # CLS token
            final_end_positions.append(0)
        else:
            # Find the index of the token that corresponds to the start and end character positions of the answer
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            final_start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            final_end_positions.append(idx + 1)

    inputs["start_positions"] = final_start_positions
    inputs["end_positions"] = final_end_positions
    return inputs


# Apply the preprocessing function to the training and validation datasets
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

#==========================
# TRAINING
#==========================

# Training settings
#data_collator = DefaultDataCollator() # show to the model how to handle the tokenized inputs # prend les questions tokenisées, les regroupe par paquets de 6 (la taille du batch) et les transforme dans le format mathématique exact (==> le modele sait comment lire les données ) 

training_args = TrainingArguments(
    output_dir="./biobert_bioasq6b_fact_list_model",
    evaluation_strategy="epoch",      
    save_strategy="epoch",       
    learning_rate=2e-5,
    per_device_train_batch_size=6, 
    per_device_eval_batch_size=6,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,           
    load_best_model_at_end=True, 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val#, 
    #data_collator=data_collator,
)

# Launch the training process
trainer.train()


# =========================
#  SAVE MODEL
# =========================

model.save_pretrained("./biobert_bioasq6b_fact_list_model")
tokenizer.save_pretrained("./biobert_bioasq6b_fact_list_model")

print("Training finished!")