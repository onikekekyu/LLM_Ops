# scripts/pipeline_runner.py

import os
import sys
from datetime import datetime

import google.cloud.aiplatform as aiplatform
from kfp import compiler

# Ajout du projet racine au path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import constants
from src.pipelines.model_training_pipeline import model_training_pipeline

PIPELINE_JSON_FILE = "yoda_finetuning_pipeline.json"

def main():
    """Compile et exécute la pipeline sur Vertex AI."""

    compiler.Compiler().compile(
        pipeline_func=model_training_pipeline,
        package_path=PIPELINE_JSON_FILE,
    )
    print(f"Pipeline compilée avec succès en '{PIPELINE_JSON_FILE}'.")

    aiplatform.init(
        project=constants.GCP_PROJECT_ID,
        location=constants.GCP_REGION,
    )
    print("Client Vertex AI initialisé.")

    # Créer un nom unique pour chaque exécution
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    display_name = f"yoda-finetuning-run-{timestamp}"

    pipeline_job = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=PIPELINE_JSON_FILE,
        pipeline_root=constants.PIPELINE_ROOT,
        parameter_values={
            # Assurez-vous que ce chemin est correct
            "raw_data_gcs_path": "gs://bucket-llm-ops/yoda_sentences.csv"
        },
        enable_caching=True, # Réactiver le cache est une bonne pratique
    )
    
    print("Soumission de la pipeline à Vertex AI...")
    pipeline_job.submit()
    print("Pipeline soumise avec succès !")
    print(f"Suivez l'exécution ici : {pipeline_job._dashboard_uri()}")

if __name__ == "__main__":
    main()