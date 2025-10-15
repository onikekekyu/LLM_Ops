# src/pipelines/model_training_pipeline.py

from kfp import dsl
from src import constants
from src.pipeline_components.data_transformation_component import process_yoda_data

@dsl.pipeline(
    name="yoda-finetuning-pipeline",
    description="Pipeline pour préparer les données et fine-tuner un modèle.",
    pipeline_root=constants.PIPELINE_ROOT
)
def model_training_pipeline(
    # Le chemin est maintenant plus précis, correspondant à votre runner
    raw_data_gcs_path: str = "gs://bucket-llm-ops/yoda-sentences.csv"
):
    """Définit le workflow de la pipeline."""
    
    # L'appel est le même, mais il est bon de savoir que `data_processing_task`
    # contient maintenant une référence aux artefacts de sortie.
    data_processing_task = process_yoda_data(
        raw_data_gcs_path=raw_data_gcs_path
    )

    # Pour une étape suivante, vous accéderiez aux sorties comme ceci :
    # training_task = train_model(
    #     train_data=data_processing_task.outputs["train_dataset"]
    # )