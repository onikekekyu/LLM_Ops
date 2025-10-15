import logging
from kfp import dsl
from typing import NamedTuple

# Configuration du logging
logging.basicConfig(level=logging.INFO)

# Définition du composant avec le décorateur @component
# On spécifie l'image de base et les paquets à installer dans le conteneur
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
) -> NamedTuple(
    "Outputs",
    [
        ("train_dataset_path", str),
        ("test_dataset_path", str),
    ],
):
    """
    Lit les données brutes depuis GCS, les formate pour Phi-3,
    les divise en ensembles d'entraînement et de test, et les sauvegarde sur GCS.
    """
    logging.info(f"Lecture du dataset depuis : {raw_data_gcs_path}")
    # Import heavy dependencies inside the function so module import doesn't fail in dev.
    import pandas as pd
    from datasets import Dataset

    # Pandas lit directement depuis un chemin GCS
    df = pd.read_csv(raw_data_gcs_path)
    
    # Conversion en objet Dataset de Hugging Face
    hf_dataset = Dataset.from_pandas(df)
    logging.info("Conversion en Dataset Hugging Face réussie.")

    # Fonction pour appliquer le template de chat
    def format_chat_template(example):
        return {
            "messages": [
                {"role": "user", "content": example["english_sentence"]},
                {"role": "assistant", "content": example["yoda_sentence"]},
            ]
        }

    # Appliquer le formatage
    formatted_dataset = hf_dataset.map(format_chat_template)
    logging.info("Formatage du dataset au format conversationnel terminé.")

    # Diviser le dataset (80% train, 20% test)
    split_dataset = formatted_dataset.train_test_split(test_size=0.2, seed=42)
    train_split = split_dataset["train"]
    test_split = split_dataset["test"]
    logging.info(f"Dataset divisé : {len(train_split)} exemples d'entraînement, {len(test_split)} exemples de test.")

    # Define output paths
    train_output_path = "/tmp/train.csv"
    test_output_path = "/tmp/test.csv"

    # Sauvegarder les datasets
    train_split.to_csv(train_output_path, index=False)
    test_split.to_csv(test_output_path, index=False)
    logging.info(f"Dataset d'entraînement sauvegardé sur : {train_output_path}")
    logging.info(f"Dataset de test sauvegardé sur : {test_output_path}")
    logging.info("✅ Tâche terminée avec succès !")

    return (train_output_path, test_output_path)