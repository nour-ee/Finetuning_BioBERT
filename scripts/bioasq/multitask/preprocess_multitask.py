import json
import re
import os
import pandas as pd
from datasets import Dataset

def load_and_process_multitask_bioasq(filepath, is_training=True):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    multitask_records = []
    yesno_count = 0
    qa_count = 0
    
    for q in data['questions']:
        if q['type'] not in ['factoid', 'list', 'yesno']:
            continue
            
        # As-Snippets-is Strategy : each snippet is treated as a separate context for the same question
        for snippet in q.get('snippets', []):
            snippet_text = snippet.get('text', '').strip()
            context_normalized = re.sub(r'\s+', ' ', snippet_text).strip()
            
            if not context_normalized:
                continue
            
            # --- QUESTIONS YES/NO (task_id = 1) ---
            if q['type'] == 'yesno':
                raw_answer = str(q.get('exact_answer', '')).lower()
                label = 1 if 'yes' in raw_answer else 0
                
                multitask_records.append({
                    "id": q['id'],
                    "task_id": 1,  
                    "type": q['type'],
                    "context": context_normalized,
                    "question": q['body'].strip(),
                    "label": label,
                    "answers": {"text": [], "answer_start": []} 
                })
                yesno_count += 1
                
            # --- QUESTIONS FACTOID & LIST (task_id = 0) ---
            elif q['type'] in ['factoid', 'list']:
                exact_answers = q.get('exact_answer', [])
                if not exact_answers:
                    continue
                    
                if isinstance(exact_answers[0], list):
                    answers_list = [item for sublist in exact_answers for item in sublist]
                else:
                    answers_list = exact_answers

                valid_texts = []
                valid_starts = []
                
                for ans in answers_list:
                    ans_clean = re.sub(r'\s+', ' ', str(ans).strip())
                    pos = context_normalized.find(ans_clean)
                    
                    if pos == -1:
                        match = re.search(re.escape(ans_clean), context_normalized, re.IGNORECASE)
                        if match:
                            pos = match.start()
                            ans_clean = match.group()
                    
                    if pos != -1: 
                        valid_texts.append(ans_clean)
                        valid_starts.append(pos)
                
                if not valid_texts:
                    continue 
                    
                multitask_records.append({
                    "id": q['id'],
                    "task_id": 0,  # Flag d'identification de tâche explicite
                    "type": q['type'],
                    "context": context_normalized,
                    "question": q['body'].strip(),
                    "label": -1,   # Label ignoré pour la tête classification
                    "answers": {
                        "text": valid_texts,
                        "answer_start": valid_starts
                    }
                })
                qa_count += 1
                
    df = pd.DataFrame(multitask_records)
    
    # Undersampling for the training set to balance Yes/No classes 
    if is_training and not df.empty:
        df_yn = df[df['task_id'] == 1]
        df_qa = df[df['task_id'] == 0]
        
        if not df_yn.empty and not df_qa.empty:
            counts = df_yn['label'].value_counts()
            if len(counts) == 2:
                min_class = counts.min()
                # Balance Yes/No classes by undersampling the majority class (e.g., Yes) to match the minority class (e.g., No)
                df_yn_balanced = df_yn.groupby('label').apply(lambda x: x.sample(n=min_class, random_state=42)).reset_index(drop=True)
                df = pd.concat([df_qa, df_yn_balanced], ignore_index=True)

    return Dataset.from_pandas(df)

if __name__ == "__main__":
    os.makedirs("datasets", exist_ok=True)

    print("\n--- PREPROCESSING DATASETS ---")
    
    # Path to your official BioASQ training file (e.g., training13b.json)
    train_dataset_raw = load_and_process_multitask_bioasq(
        'datasets/BioASQ-training13b/BioASQ-training13b/training13b.json', 
        is_training=True
    )
    
    # Path to your first golden validation file (e.g., 13B1_golden.json)
    val_dataset_raw = load_and_process_multitask_bioasq(
        'datasets/Task6BGoldenEnriched/13B1_golden.json', 
        is_training=False
    )
    
    # Save the processed datasets in JSON files
    train_dataset_raw.to_json("datasets/multitask_bioasq_train13b.json", orient="records", lines=True)
    val_dataset_raw.to_json("datasets/multitask_bioasq_val13b.json", orient="records", lines=True)

    print("The training dataset is saved as `multitask_bioasq_train13b.json` and the validation dataset as `multitask_bioasq_val13b.json` in datasets directory.")
