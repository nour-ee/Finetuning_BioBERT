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
        
        # 1. Load my specialized Factoid/List Extraction Model
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_path)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_path).to(self.device)
        
        # 2. Load my specialized Yes/No Classification Model
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
        # ROUTING : FACTOID & LIST (Stratégie As-Snippets + Seuil du Papier)
        # =====================================================================
        if question_type in ['factoid', 'list']:
            all_candidates = {}

            # # Tenter d'extraire le nombre de réponses attendues (ex: "list 6 symptoms")
            # match_number = re.search(r'\b(list|give|name)\s+(\d+)\b', question.lower())
            # expected_count = int(match_number.group(2)) if match_number else None

            # Étape 1 : On boucle sur CHAQUE snippet individuellement (Inférence As-Snippets)
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

                # Calcul des probabilités par softmax sur les logits pour appliquer le seuil
                start_probs = torch.softmax(outputs.start_logits, dim=-1)[0].cpu().numpy()
                end_probs = torch.softmax(outputs.end_logits, dim=-1)[0].cpu().numpy()

                # On extrait les 5 meilleurs index de début et de fin pour explorer les combinaisons
                n_best = 5
                start_indexes = np.argsort(start_probs)[::-1][:n_best]
                end_indexes = np.argsort(end_probs)[::-1][:n_best]

                for start_idx in start_indexes:
                    for end_idx in end_indexes:
                        # Filtrer les index invalides ou inversés
                        if start_idx >= len(start_probs) or end_idx >= len(end_probs) or start_idx > end_idx:
                            continue
                        
                        # Ignorer si la réponse est trop longue (plus de 15 tokens)
                        if end_idx - start_idx + 1 > 15:
                            continue

                        # Calcul de la probabilité combinée du span (P_start * P_end)
                        span_prob = start_probs[start_idx] * end_probs[end_idx]

                        # Décodage du texte de la réponse
                        all_tokens = self.qa_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                        ans_text = self.qa_tokenizer.convert_tokens_to_string(all_tokens[start_idx : end_idx + 1]).strip()

                        # Ignorer les tokens spéciaux récupérés par erreur
                        if not ans_text or "[CLS]" in ans_text or "[SEP]" in ans_text:
                            continue
                        
                        # Nettoyage des bords et filtrage des parenthèses incomplètes (Section 2.4 du papier)
                        ans_text = ans_text.strip(',.()').strip()
                        if ans_text.count('(') != ans_text.count(')'):
                            continue

                        # Stocker ou mettre à jour la probabilité maximale trouvée pour ce texte exact
                        if ans_text and (ans_text not in all_candidates or span_prob > all_candidates[ans_text]):
                            all_candidates[ans_text] = span_prob

            if not all_candidates:
                return "No extraction found"

            # Tri des candidats par probabilité décroissante
            sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)

            # --- RÈGLE POUR LES FACTOIDS ---
            if question_type == 'factoid':
                return sorted_candidates[0][0]

            # --- RÈGLE POUR LES LISTES ---
            elif question_type == 'list':
                # Application stricte du seuil de validation de 0.42 du papier
                final_list = [text for text, prob in sorted_candidates if prob >= 0.42]

                # Sécurité : Si rien ne dépasse le seuil, on renvoie au moins le Top 1 global
                if not final_list:
                    final_list = [sorted_candidates[0][0]]

                # Retourne la liste délimitée par un séparateur standard (ex: " | ")
                return " | ".join(final_list)

        # =====================================================================
        # ROUTING : YES/NO (Stratégie As-Passages - Concaténation Complète)
        # =====================================================================
        elif question_type == 'yesno':
            # Pour Yes/No, on fusionne absolument tout le contexte pour la classification globale
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
    # Point de repère vers tes dossiers de modèles entraînés
    pipeline = BioASQRoutingPipeline(
        qa_model_path="./biobert_bioasq13b_fact_list_model", 
        yesno_model_path="./biobert_bioasq13b_yesno_model"
    )
    
    # Cibler tes fichiers golden bruts 
    test_files = glob.glob("datasets/Task13BGoldenEnriched/Task13BGoldenEnriched/13B[2-4]_golden.json")
    
    all_dfs = []
    
    for f_path in test_files:
        name, df_predictions = pipeline.evaluate_raw_golden_file(f_path)
        df_predictions["Source File"] = name
        all_dfs.append(df_predictions)
    
    # Concaténation de toutes les prédictions dans un export CSV global unique
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # S'assurer que le dossier results existe
        os.makedirs("./results", exist_ok=True)
        output_csv = "./results/pipeline_predictions_combined_13B2_13B4.csv"
        combined_df.to_csv(output_csv, index=False)
        
        print("\n" + "="*50)
        print(f"Extraction terminée avec succès !")
        print(f"Fichier global combiné sauvegardé sous : {output_csv}")
        print(f"Nombre total de questions traitées : {len(combined_df)}")
        print("="*50)
    else:
        print("Aucun fichier trouvé. Vérifie le pattern de ton chemin glob.")