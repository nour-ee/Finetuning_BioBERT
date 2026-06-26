import json
import torch
import re
import os
import glob
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from scripts.bioasq.paper_approach.train_multitask import MultitaskBioBERT 

class UnifiedBioASQPipeline:
    def __init__(self, model_weights_path, tokenizer_path):
        print("Initializing Unified Multitask Pipeline...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the tokenizer from the specified path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load the multitask model and inject the optimized weights
        self.model = MultitaskBioBERT()
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Unified Multitask Model deployed on: {self.device}")

    def _clean_text(self, text):
        return " ".join(text.strip().split())

    def predict_single(self, question_type, question_body, snippets_list):
        question = question_body.strip()
        
        if not snippets_list:
            return "Error: Empty Context snippets."

        # =====================================================================
        # TASK 0 : ROUTING FACTOID & LIST (AS-SNIPPET-IS STRATEGY + THRESHOLD 0.42)
        # =====================================================================
        if question_type in ['factoid', 'list']:
            all_candidates = {}

            #  Independent inference for EACH snippet (As-Snippet Strategy)
            for snippet in snippets_list:
                context = self._clean_text(snippet.get('text', '').strip())
                if not context:
                    continue

                inputs = self.tokenizer(
                    question, 
                    context, 
                    max_length=384, 
                    truncation="only_second", 
                    return_tensors="pt",
                    padding="max_length"
                ).to(self.device)

                # Manual injection of the task flag for the QA head
                inputs['task_ids'] = torch.tensor([0], device=self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Calculate the probabilities for start and end positions using softmax
                start_probs = torch.softmax(outputs["start_logits"], dim=-1)[0].cpu().numpy()
                end_probs = torch.softmax(outputs["end_logits"], dim=-1)[0].cpu().numpy()

                # Exploration of the Top 5 to avoid truncation and handle lists
                n_best = 5
                start_indexes = np.argsort(start_probs)[::-1][:n_best]
                end_indexes = np.argsort(end_probs)[::-1][:n_best]

                for start_idx in start_indexes:
                    for end_idx in end_indexes:
                        if start_idx >= len(start_probs) or end_idx >= len(end_probs) or start_idx > end_idx:
                            continue

                        # Calculate the combined probability of the span
                        span_prob = start_probs[start_idx] * end_probs[end_idx]

                        # Decode the span text from the token IDs
                        all_tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                        ans_text = self.tokenizer.convert_tokens_to_string(all_tokens[start_idx : end_idx + 1]).strip()

                        if not ans_text or "[CLS]" in ans_text or "[SEP]" in ans_text:
                            continue
                        
                        # Post-processing: clean and filter asymmetric parentheses
                        ans_text = ans_text.strip(',.()').strip()
                        if ans_text.count('(') != ans_text.count(')'):
                            continue

                        # Save the best probability for this exact string
                        if ans_text and (ans_text not in all_candidates or span_prob > all_candidates[ans_text]):
                            all_candidates[ans_text] = span_prob

            if not all_candidates:
                return "No extraction found"

            # Sorted candidates by descending probability
            sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)

            # --- For factoid, we return the top candidate---
            if question_type == 'factoid':
                return sorted_candidates[0][0]

            # --- For list, we apply a strict threshold of 0.42 ---
            elif question_type == 'list':
                final_list = [text for text, prob in sorted_candidates if prob >= 0.42]

                # Security: return at least the Top 1 if nothing exceeds the threshold
                if not final_list:
                    final_list = [sorted_candidates[0][0]]

                return " | ".join(final_list)

        # =====================================================================
        # TASK 1 : ROUTING YES/NO (Stratégie As-Passages / global Concatenation )
        # =====================================================================
        elif question_type == 'yesno':
            # context is the concatenation of all snippets
            context = self._clean_text(" ".join([s.get('text', '').strip() for s in snippets_list]))
            
            inputs = self.tokenizer(
                question, 
                context, 
                max_length=384, 
                truncation="only_second", 
                return_tensors="pt", 
                padding="max_length"
            ).to(self.device)
            
            # Manual injection of the task flag for the classification head
            inputs['task_ids'] = torch.tensor([1], device=self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
            
            prediction = torch.argmax(outputs["classification_logits"], dim=-1).item()
            return "yes" if prediction == 1 else "no"
            
        else:
            return f"Skipped: Type '{question_type}' not natively handled."

    def evaluate_raw_golden_file(self, json_filepath):
        filename = os.path.basename(json_filepath)
        print(f"Processing evaluation collection: {filename}")
        
        with open(json_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        results = []
        for q in data['questions']:
            pred_answer = self.predict_single(q['type'], q['body'], q.get('snippets', []))
            gold_ans = q.get('exact_answer', "N/A")
            
            results.append({
                "Question ID": q['id'],
                "Question Type": q['type'],
                "Question Body": q['body'],
                "Ground Truth Answer": gold_ans,
                "Pipeline Predicted Answer": pred_answer
            })
            
        return filename, pd.DataFrame(results)

# =====================================================================
# EXECUTION
# =====================================================================
if __name__ == "__main__":
    pipeline = UnifiedBioASQPipeline(
        model_weights_path="./multitask_biobert_model.pt",
        tokenizer_path="./multitask_biobert_tokenizer"
    )
    
    test_files = glob.glob("datasets/Task6BGoldenEnriched/6B[2-5]_golden.json")
    all_dfs = []
    
    for f_path in test_files:
        name, df_predictions = pipeline.evaluate_raw_golden_file(f_path)
        df_predictions["Source File"] = name
        all_dfs.append(df_predictions)
        
    if all_dfs:
        os.makedirs("./results", exist_ok=True)
        combined_df = pd.concat(all_dfs, ignore_index=True)
        output_csv = "./results/multitask_pipeline_predictions_6B_combined.csv"
        combined_df.to_csv(output_csv, index=False)
        print(f"\nExecution finished! Saved combined output predictions to: {output_csv}")
