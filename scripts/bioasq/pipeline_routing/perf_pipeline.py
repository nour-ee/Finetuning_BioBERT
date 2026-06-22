import pandas as pd
import ast
import re
import os

def clean_eval_string(text):
    """Nettoie le texte médical en supprimant la ponctuation et les espaces superflus."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Supprime les résidus visuels de tokenisation (espaces autour des tirets/virgules)
    text = re.sub(r'\s*([-,.\/])\s*', r'\1', text)
    # Enlève la ponctuation pour ne garder que les caractères alphanumériques et espaces
    text = re.sub(r'[^\w\s]', '', text)
    return " ".join(text.split())

def extraire_options_valides(raw_truth):
    """
    Découpe et extrait proprement toutes les chaînes de texte valides
    à partir de la vérité terrain, peu importe le niveau d'imbrication des crochets.
    """
    text_raw = str(raw_truth).strip()
    if text_raw == "N/A" or not text_raw:
        return []
        
    # Extraction magique : trouve tout ce qui est entre guillemets simples ou douches
    options = re.findall(r"['\"]([^'\"]+)['\"]", text_raw)
    
    # Si la regex ne trouve rien (pas de guillemets), on nettoie la chaîne brute
    if not options:
        cleaned = text_raw.replace('[', '').replace(']', '').strip()
        if cleaned:
            options = [cleaned]
            
    return [clean_eval_string(opt) for opt in options if opt]

def calculer_performances_pipeline(csv_filepath, output_csv="./results/pipeline_metrics_summary.csv"):
    df = pd.read_csv(csv_filepath)
    
    # On élimine les lignes de fallback ou d'erreur
    df = df[~df["Pipeline Predicted Answer"].str.contains("fallback|Skipped|Error", na=False, case=False)]
    
    print(f"Évaluation sur {len(df)} questions valides (Yes/No, Factoid, List)...")
    
    metrics_data = {"Metric Name": [], "Question Type": [], "Value": []}
    
    # ==========================================
    # ÉVALUATION YES/NO
    # ==========================================
    df_yn = df[df["Question Type"] == "yesno"]
    if not df_yn.empty:
        y_true = df_yn["Ground Truth Answer"].apply(lambda x: str(x).lower().strip("[]'\" "))
        y_pred = df_yn["Pipeline Predicted Answer"].str.lower().str.strip()
        accuracy_yn = (y_true == y_pred).mean()
        
        print(f"\n--- PERFORMANCE YES/NO (Total: {len(df_yn)}) ---")
        print(f"Accuracy : {accuracy_yn * 100:.2f}%")
        metrics_data["Metric Name"].append("Accuracy")
        metrics_data["Question Type"].append("yesno")
        metrics_data["Value"].append(accuracy_yn)
    
    # ==========================================
    # ÉVALUATION FACTOID
    # ==========================================
    df_fact = df[df["Question Type"] == "factoid"]
    if not df_fact.empty:
        exact_matches = 0
        for _, row in df_fact.iterrows():
            pred = clean_eval_string(row["Pipeline Predicted Answer"])
            if "no extraction found" in pred or not pred:
                continue
                
            # Extraction propre des synonymes acceptés
            truths = extraire_options_valides(row["Ground Truth Answer"])
            
            # Si la prédiction matche l'une des vérités ou inversement
            if any(pred in t or t in pred for t in truths if t):
                exact_matches += 1
                
        strict_accuracy = exact_matches / len(df_fact)
        print(f"\n--- PERFORMANCE FACTOID (Total: {len(df_fact)}) ---")
        print(f"Strict Accuracy (Flexible Match) : {strict_accuracy * 100:.2f}%")
        metrics_data["Metric Name"].append("Strict Accuracy (EM)")
        metrics_data["Question Type"].append("factoid")
        metrics_data["Value"].append(strict_accuracy)

    # ==========================================
    # ÉVALUATION LIST (Corrigée pour le format multi-réponses " | ")
    # ==========================================
    df_list = df[df["Question Type"] == "list"]
    if not df_list.empty:
        total_precision = 0
        total_recall = 0
        
        for _, row in df_list.iterrows():
            raw_pred = str(row["Pipeline Predicted Answer"]).strip()
            if not raw_pred or "no extraction found" in raw_pred.lower():
                continue
                
            truths = extraire_options_valides(row["Ground Truth Answer"])
            
            # Découper la chaîne par le pipe "|" et nettoyer chaque entité extraite
            preds_list = [clean_eval_string(p) for p in raw_pred.split("|")]
            preds_list = [p for p in preds_list if p]  # Filtrer les chaînes vides résiduelles
            
            if not preds_list:
                continue
            
            true_positives = 0
            # On vérifie la correspondance de chaque prédiction par rapport aux vérités
            for p in preds_list:
                if any(p in t or t in p for t in truths if t):
                    true_positives += 1
            
            # Calcul des métriques par question
            precision = true_positives / len(preds_list) if len(preds_list) > 0 else 0
            recall = true_positives / max(1, len(truths))
            
            total_precision += precision
            total_recall += recall
            
        mean_precision = total_precision / len(df_list)
        mean_recall = total_recall / len(df_list)
        f1_list = (2 * mean_precision * mean_recall) / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0
        
        print(f"\n--- PERFORMANCE LIST (Total: {len(df_list)}) ---")
        print(f"Precision moyenne : {mean_precision * 100:.2f}%")
        print(f"Recall moyen      : {mean_recall * 100:.2f}%")
        print(f"F1-Score moyen    : {f1_list * 100:.2f}%")
        
        metrics_data["Metric Name"].extend(["Precision", "Recall", "F1-Score"])
        metrics_data["Question Type"].extend(["list", "list", "list"])
        metrics_data["Value"].extend([mean_precision, mean_recall, f1_list])

    # ==========================================
    # SAUVEGARDE DU RAPPORT
    # ==========================================
    if metrics_data["Metric Name"]:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics["Value (%)"] = df_metrics["Value"] * 100
        df_metrics = df_metrics.drop(columns=["Value"])
        df_metrics.to_csv(output_csv, index=False)
        print(f"\n[INFO] Rapport sauvegardé sous : {output_csv}")

if __name__ == "__main__":
    # Relance automatique sur ton fichier combiné
    input_file = "./results/pipeline_predictions_combined_13B2_13B4.csv"
    if os.path.exists(input_file):
        calculer_performances_pipeline(input_file)
    else:
        print(f"[ERREUR] Le fichier de prédictions {input_file} n'existe pas. Génère-le d'abord.")