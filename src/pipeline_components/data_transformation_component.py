# src/pipeline_components/data_transformation_component.py

import logging
from kfp import dsl

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dsl.component(
    base_image="python:3.10-slim",
    packages_to_install=[
        "pandas==2.2.2",
        "datasets==2.19.0",
        "gcsfs==2024.3.1",
        "pyarrow==16.1.0",
    ],
)
def process_yoda_data(
    raw_data_gcs_path: str,
    # MODIFIÉ : On déclare les sorties comme des Artefacts de type 'Dataset'
    # KFP fournira automatiquement un chemin GCS pour ces variables.
    train_dataset: dsl.OutputPath(dsl.Dataset),
    test_dataset: dsl.OutputPath(dsl.Dataset),
):
    """
    Lit les données brutes depuis GCS, les formate pour Phi-3,
    les divise en ensembles d'entraînement et de test, et les sauvegarde sur GCS.
    """
    # Importations à l'intérieur pour alléger l'import du module
    import logging
    import pandas as pd
    from datasets import Dataset

    logging.info(f"Lecture du dataset depuis : {raw_data_gcs_path}")
    df = pd.read_csv(raw_data_gcs_path)
    
    # S'assurer que les colonnes existent
    if "sentence" not in df.columns or "translation" not in df.columns:
        raise ValueError("Le CSV doit contenir les colonnes 'sentence' et 'translation'.")

    hf_dataset = Dataset.from_pandas(df)
    logging.info("Conversion en Dataset Hugging Face réussie.")

    def format_chat_template(example):
        # MODIFIÉ : Utilisation des bons noms de colonnes du CSV
        return {
            "messages": [
                {"role": "user", "content": example["sentence"]},
                {"role": "assistant", "content": example["translation"]},
            ]
        }

    # Supprimer les colonnes originales après le formatage pour ne garder que "messages"
    formatted_dataset = hf_dataset.map(format_chat_template, remove_columns=hf_dataset.column_names)
    logging.info("Formatage du dataset au format conversationnel terminé.")

    split_dataset = formatted_dataset.train_test_split(test_size=0.2, seed=42)
    train_split = split_dataset["train"]
    test_split = split_dataset["test"]
    logging.info(f"Dataset divisé : {len(train_split)} exemples d'entraînement, {len(test_split)} exemples de test.")

    # MODIFIÉ : Sauvegarder les datasets en utilisant les chemins fournis par KFP
    # Nous utilisons to_json pour mieux préserver la structure conversationnelle.
    train_split.to_json(train_dataset)
    test_split.to_json(test_dataset)
    
    logging.info(f"Dataset d'entraînement sauvegardé sur : {train_dataset}")
    logging.info(f"Dataset de test sauvegardé sur : {test_dataset}")
    logging.info("✅ Tâche terminée avec succès !")

    # MODIFIÉ : Plus besoin de retourner les chemins, KFP les gère automatiquement.
    # La fonction peut ne rien retourner.