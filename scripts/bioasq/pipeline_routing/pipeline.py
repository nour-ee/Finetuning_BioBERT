import json
import torch
import re
import os
import pandas as pd
import glob
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification

class BioASQRoutingPipeline:
    def __init__(self, qa_model_path, yesno_model_path):
        print("Initializing Routing Pipeline and loading models...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the Factoid/List Extraction Model
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_path)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_path).to(self.device)
        
        # Load the Yes/No Classification Model
        self.yn_tokenizer = AutoTokenizer.from_pretrained(yesno_model_path)
        self.yn_model = AutoModelForSequenceClassification.from_pretrained(yesno_model_path).to(self.device)
        
        # Set both models to evaluation mode immediately for inference
        self.qa_model.eval()
        self.yn_model.eval()
        print(f"Models successfully deployed on device: {self.device}")

    def _clean_text(self, text):
        return " ".join(text.strip().split())

    def predict_single(self, question_type, question_body, snippets_list):
        """Inspects the type and dynamically routes inputs to the correct model and strategy."""
        question = question_body.strip()
        
        if not snippets_list:
            return "Error: Empty Context snippets."

        # =====================================================================
        # ROUTING : FACTOID & LIST (As-Snippets-Is Strategy + THRESHOLDING)
        # =====================================================================
        if question_type in ['factoid', 'list']:
            all_candidates = {}

            # Loop through each snippet individually (As-Snippets-is Strategy)
            for snippet in snippets_list:
                context = self._clean_text(snippet.get('text', '').strip())
                if not context:
                    continue

                inputs = self.qa_tokenizer(
                    question, 
                    context, 
                    max_length=384, 
                    truncation="only_second", 
                    return_tensors="pt",
                    padding="max_length"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.qa_model(**inputs)

                # Calculate the probabilities using softmax on the logits to apply thresholding
                start_probs = torch.softmax(outputs.start_logits, dim=-1)[0].cpu().numpy()
                end_probs = torch.softmax(outputs.end_logits, dim=-1)[0].cpu().numpy()

                # We extract the top 5 start and end indices to explore combinations
                n_best = 5
                start_indexes = np.argsort(start_probs)[::-1][:n_best]
                end_indexes = np.argsort(end_probs)[::-1][:n_best]

                for start_idx in start_indexes:
                    for end_idx in end_indexes:
                        # Filter the invalid or reversed indices
                        if start_idx >= len(start_probs) or end_idx >= len(end_probs) or start_idx > end_idx:
                            continue

                        # Decode the answer text from the token indices
                        all_tokens = self.qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                        ans_text = self.qa_tokenizer.convert_tokens_to_string(all_tokens[start_idx : end_idx + 1]).strip()

                        # Ignore special tokens 
                        if not ans_text or "[CLS]" in ans_text or "[SEP]" in ans_text:
                            continue
                        
                        # Cleaning the answer text by stripping unwanted characters and ensuring balanced parentheses
                        ans_text = ans_text.strip(',.()').strip()
                        if ans_text.count('(') != ans_text.count(')'):
                            continue

                        # Storage or update the maximum probability found for this exact text
                        if ans_text and (ans_text not in all_candidates or span_prob > all_candidates[ans_text]):
                            all_candidates[ans_text] = span_prob

            if not all_candidates:
                return "No extraction found"

            # Sort the candidates by their probabilities in descending order
            sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)

            # --- RULE FOR FACTOIDS : HIGHEST PROBABILITY ---
            if question_type == 'factoid':
                return sorted_candidates[0][0]

            # --- RULE FOR LISTS : BY THRESHOLDING (0.42) ---
            
            elif question_type == 'list':
                # Application stricte du seuil de validation de 0.42 du papier
                final_list = [text for text, prob in sorted_candidates if prob >= 0.42]

                # Security: If nothing exceeds the threshold, return at least the Top 1 global
                if not final_list:
                    final_list = [sorted_candidates[0][0]]

                # Returning the list of answers as a single string separated by " | "
                return " | ".join(final_list)

        # ===================
        # ROUTING : YES/NO 
        # ===================
        
        elif question_type == 'yesno':
            # We concatenate all snippet texts into a single context string for the yes/no classification
            context = self._clean_text(" ".join([s.get('text', '').strip() for s in snippets_list]))
            inputs = self.yn_tokenizer(question, context, max_length=384, truncation="only_second", return_tensors="pt", padding="max_length").to(self.device)
            
            with torch.no_grad():
                outputs = self.yn_model(**inputs)
            
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            return "yes" if prediction == 1 else "no"
            
        elif question_type == 'summary':
            return "Summary routing fallback: Requires an abstractive generative decoder."
            
        else:
            return f"Skipped: Type '{question_type}' not supported."

    def evaluate_raw_golden_file(self, json_filepath):
        """Processes an unaltered golden file, routing every single question without pre-exclusion."""
        filename = os.path.basename(json_filepath)
        print(f"\nProcessing test collection file: {filename}")
        
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

# ==============================================
# PIPELINE EXECUTION (EVALUATION ON GOLDEN FILES)
# ==============================================
if __name__ == "__main__":
    pipeline = BioASQRoutingPipeline(
        qa_model_path="./biobert_bioasq13b_fact_list_model", 
        yesno_model_path="./biobert_bioasq13b_yesno_model"
    )
    
    test_files = glob.glob("datasets/Task13BGoldenEnriched/Task13BGoldenEnriched/13B[2-4]_golden.json")
    
    all_dfs = []
    
    for f_path in test_files:
        name, df_predictions = pipeline.evaluate_raw_golden_file(f_path)
        df_predictions["Source File"] = name
        all_dfs.append(df_predictions)
    
    # Save the combined DataFrame to a CSV file in the results directory
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # S'assurer que le dossier results existe
        os.makedirs("./results", exist_ok=True)
        output_csv = "./results/pipeline_predictions_combined_13B2_13B4.csv"
        combined_df.to_csv(output_csv, index=False)
        
        print("\n" + "="*50)
        print(f"Extraction finished successfully! Results saved in the 'results' directory.")
        print(f"csv file saved in : {output_csv}")
        print(f"Number of questions : {len(combined_df)}")
        print("="*50)
    else:
        print("No files found. Check the pattern of your glob path.")
        
