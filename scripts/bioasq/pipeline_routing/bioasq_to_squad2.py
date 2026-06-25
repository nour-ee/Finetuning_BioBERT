import json
import re
import pandas as pd
from datasets import Dataset
import os

########### PREPROCESS FACTOID & LIST QUESTIONS (AS-SNIPPETS-IS STRATEGY) ############
def load_and_process_bioasq(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    debug_filepath = f"debug_{base_name}.txt"
    
    with open(debug_filepath, "w", encoding="utf-8") as debug_file:
        debug_file.write(f"=== REPORT DEBUG : {filepath} ===\n")
        debug_file.write("List of snippets/questions excluded because NO answer was found.\n")
        debug_file.write("=" * 80 + "\n\n")
        
    processed_records = []
    count_factoid = 0
    count_list = 0
    
    for q in data['questions']:
        if q['type'] in ['summary', 'yesno']:
            continue
            
        exact_answers = q.get('exact_answer', [])
        if not exact_answers:
            continue
            
        # Flatten the list of answers if necessary
        if isinstance(exact_answers[0], list):
            answers_list = [item for sublist in exact_answers for item in sublist]
        else:
            answers_list = exact_answers

        # Counting the number of factoid and list questions in the original dataset
        if q['type'] == 'factoid':
            count_factoid += 1
        elif q['type'] == 'list':
            count_list += 1

        snippets = q.get('snippets', [])
        question_matched_at_least_once = False
        

        # AS-SNIPPETS-IS STRATEGY: Treat EACH snippet as an independent context
        for snippet_idx, snippet in enumerate(snippets):
            snippet_text = snippet.get('text', '').strip()
            if not snippet_text:
                continue
                
            # Normalization of spaces for the current snippet
            snippet_normalized = re.sub(r'\s+', ' ', snippet_text).strip()
            if not snippet_normalized:
                continue
                
            valid_texts = []
            valid_starts = []
            
            # Retrieve all possible answers in THIS specific snippet
            for ans in answers_list:
                ans_clean = str(ans).strip() 
                ans_clean = re.sub(r'\s+', ' ', ans_clean)
                

                # We first try to find the answer in the snippet with exact matching
                pos = snippet_normalized.find(ans_clean)
                
                # If the answer is not found, we try to find it in a case-insensitive manner
                if pos == -1:
                    match = re.search(re.escape(ans_clean), snippet_normalized, re.IGNORECASE)
                    if match:
                        pos = match.start()
                        ans_clean = match.group()
                
                # If the answer is present in this snippet, we add it
                if pos != -1:
                    valid_texts.append(ans_clean)
                    valid_starts.append(pos)
            
            # If the snippet contains at least one answer, we add it as a distinct example
            if valid_texts:
                question_matched_at_least_once = True
                processed_records.append({
                    # Modify the ID to identify the original snippet (e.g., "5c52f01a_0")
                    "id": f"{q['id']}_{snippet_idx}", 
                    "original_id": q['id'],
                    "type": q['type'],
                    "context": snippet_normalized,
                    "question": q['body'],
                    "answers": {
                        "text": valid_texts,
                        "answer_start": valid_starts
                    }
                })
                
        # If none of the snippets for this question matched any answer, we log it for debugging
        if not question_matched_at_least_once:
            with open("debug_unmatched_questions_with_all_answersn.txt", "a", encoding="utf-8") as debug_file:
                debug_file.write(f"Question ID: {q['id']}\n")
                debug_file.write(f"Question: {q['body']}\n")
                debug_file.write(f"Expected Answers: {answers_list}\n")
                debug_file.write(f"Total Snippets Evaluated: {len(snippets)}\n")
                debug_file.write("-" * 80 + "\n")
                
    print(f"Number of target factoid questions: {count_factoid}")
    print(f"Number of target list questions: {count_list}")
    print(f"Total generated samples (as-snippets): {len(processed_records)}")
        
    return Dataset.from_pandas(pd.DataFrame(processed_records))

# ===============================================
# EXECUTION FOR TRAINING AND TESTING 
# ===============================================

# Load and process the official BioASQ training dataset (e.g., BioASQ-training13b.json)
hf_dataset = load_and_process_bioasq('datasets/BioASQ-training13b/BioASQ-training13b/training13b.json')

# Statistics of the generated dataset
factoid_count = sum(1 for example in hf_dataset if example['type'].startswith('factoid'))
list_count = sum(1 for example in hf_dataset if example['type'].startswith('list'))

print(f"Number of valid factoid snippet-samples: {factoid_count}")
print(f"Number of valid list snippet-samples: {list_count}")

output_json_path = "datasets/BioASQ_processed2_for_biobert.json"
hf_dataset.to_json(output_json_path, orient="records", lines=True)
print(f"Dataset saved successfully in : {output_json_path}\n")


##################### PREPROCESSING VALIDATION FILE ########################
    
batch_file = f"datasets/Task13BGoldenEnriched/Task13BGoldenEnriched/13B1_golden.json"
print(f"Processing {batch_file}...")
hf_dataset = load_and_process_bioasq(batch_file)

factoid_count = sum(1 for example in hf_dataset if example['type'].startswith('factoid'))
list_count = sum(1 for example in hf_dataset if example['type'].startswith('list'))
print(f"Number of valid factoid snippet-samples: {factoid_count}")
print(f"Number of valid list snippet-samples: {list_count}")
output_json_path = f"datasets/13B1_processed2.json"
hf_dataset.to_json(output_json_path, orient="records", lines=True)
print(f"Dataset saved successfully in : {output_json_path}\n")
