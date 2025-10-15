import os
import sys

import google.cloud.aiplatform as aiplatform
from kfp import compiler

# Ensure the project root is on sys.path so `src` is importable when running this script directly.
# This makes the script runnable as: python scripts/pipeline_runner.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Importer les constantes et la définition de la pipeline
from src import constants
from src.pipelines.model_training_pipeline import model_training_pipeline

# Nom du fichier YAML compilé
PIPELINE_YAML_FILE = "yoda_finetuning_pipeline.yaml"

def main():
    """Compile et exécute la pipeline sur Vertex AI."""

    # 1. Compilation de la pipeline
    compiler.Compiler().compile(
        pipeline_func=model_training_pipeline,
        package_path=PIPELINE_YAML_FILE,
    )
    print(f"Pipeline compilée avec succès en '{PIPELINE_YAML_FILE}'.")

    # 2. Initialisation du client Vertex AI
    aiplatform.init(
        project=constants.GCP_PROJECT_ID,
        location=constants.GCP_REGION,
    )
    print("Client Vertex AI initialisé.")

    # 3. Définition du Pipeline Job
    pipeline_job = aiplatform.PipelineJob(
        display_name="yoda-finetuning-pipeline-run",
        template_path=PIPELINE_YAML_FILE,
        parameter_values={
            "raw_data_gcs_path": "gs://public-datasets-bootcamp/yoda/yoda-sentences.csv"
        },
        enable_caching=False # Désactiver le cache pour les tests
    )
    print("Job de pipeline défini.")

    # 4. Soumission de la pipeline
    pipeline_job.submit()
    print("Pipeline soumise à Vertex AI.")
    print(f"Suivez l'exécution ici : {pipeline_job._dashboard_uri()}")

if __name__ == "__main__":
    main()