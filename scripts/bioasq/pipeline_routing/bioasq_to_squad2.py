import json
import re
import pandas as pd
from datasets import Dataset
import os

########### PREPROCESS FACTOID & LIST QUESTIONS (STRATÉGIE AS-SNIPPETS) ############
def load_and_process_bioasq(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    debug_filepath = f"debug_{base_name}.txt"
    
    with open(debug_filepath, "w", encoding="utf-8") as debug_file:
        debug_file.write(f"=== RAPPORT DE DÉBOGAGE POUR : {filepath} ===\n")
        debug_file.write("Liste des snippets/questions exclus car AUCUNE réponse n'a été trouvée.\n")
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
            
        # Aplatir la liste des réponses si nécessaire
        if isinstance(exact_answers[0], list):
            answers_list = [item for sublist in exact_answers for item in sublist]
        else:
            answers_list = exact_answers

        # Compteurs basés sur les questions d'origine sélectionnées
        if q['type'] == 'factoid':
            count_factoid += 1
        elif q['type'] == 'list':
            count_list += 1

        snippets = q.get('snippets', [])
        question_matched_at_least_once = False
        
        # STRATÉGIE AS-SNIPPETS : On traite CHAQUE snippet comme un contexte indépendant
        for snippet_idx, snippet in enumerate(snippets):
            snippet_text = snippet.get('text', '').strip()
            if not snippet_text:
                continue
                
            # Normalisation des espaces pour le snippet courant
            snippet_normalized = re.sub(r'\s+', ' ', snippet_text).strip()
            if not snippet_normalized:
                continue
                
            valid_texts = []
            valid_starts = []
            
            # Recherche de TOUTES les réponses possibles dans CE snippet précis
            for ans in answers_list:
                ans_clean = str(ans).strip() 
                ans_clean = re.sub(r'\s+', ' ', ans_clean)
                
                # 1. Tentative stricte
                pos = snippet_normalized.find(ans_clean)
                
                # 2. Tentative insensible à la casse
                if pos == -1:
                    match = re.search(re.escape(ans_clean), snippet_normalized, re.IGNORECASE)
                    if match:
                        pos = match.start()
                        ans_clean = match.group()
                
                # Si la réponse est présente dans ce snippet, on l'ajoute
                if pos != -1:
                    valid_texts.append(ans_clean)
                    valid_starts.append(pos)
            
            # Si le snippet contient au moins une réponse, on l'ajoute comme un exemple distinct
            if valid_texts:
                question_matched_at_least_once = True
                processed_records.append({
                    # On modifie l'ID pour identifier le snippet d'origine (ex: "5c52f01a_0")
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
                
        # Logging de debug uniquement si aucun snippet de la question n'a pu matcher
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

# =====================================================================
# EXECUTION POUR TRAINING ET TEST (Le reste de votre code reste identique)
# =====================================================================

# Extraction et processus
hf_dataset = load_and_process_bioasq('datasets/BioASQ-training13b/BioASQ-training13b/training13b.json')

# Vérification des -1
if len(hf_dataset) > 0:
    minus_one_count = sum(1 for example in hf_dataset if example['answers']['answer_start'][0] == -1)
    print(f"Number of examples with -1 (should be 0): {minus_one_count}")

# Statistiques du dataset généré
factoid_count = sum(1 for example in hf_dataset if example['type'].startswith('factoid'))
list_count = sum(1 for example in hf_dataset if example['type'].startswith('list'))

print(f"Number of valid factoid snippet-samples: {factoid_count}")
print(f"Number of valid list snippet-samples: {list_count}")

output_json_path = "datasets/BioASQ_processed2_for_biobert.json"
hf_dataset.to_json(output_json_path, orient="records", lines=True)
print(f"Dataset sauvegardé avec succès dans : {output_json_path}\n")


##################### PREPROCESSING TEST FILES ########################
for batch_num in range(1, 5):
    batch_file = f"datasets/Task13BGoldenEnriched/Task13BGoldenEnriched/13B{batch_num}_golden.json"
    print(f"Processing {batch_file}...")
    hf_dataset = load_and_process_bioasq(batch_file)
    
    if len(hf_dataset) > 0:
        minus_one_count = sum(1 for example in hf_dataset if example['answers']['answer_start'][0] == -1)
        print(f"Number of examples with -1 (should be 0): {minus_one_count}")

    factoid_count = sum(1 for example in hf_dataset if example['type'].startswith('factoid'))
    list_count = sum(1 for example in hf_dataset if example['type'].startswith('list'))

    print(f"Number of valid factoid snippet-samples: {factoid_count}")
    print(f"Number of valid list snippet-samples: {list_count}")

    output_json_path = f"datasets/13B{batch_num}_processed2.json"
    hf_dataset.to_json(output_json_path, orient="records", lines=True)
    print(f"Dataset sauvegardé avec succès dans : {output_json_path}\n")