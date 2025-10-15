from kfp import dsl
from src import constants
from src.pipeline_components.data_transformation_component import process_yoda_data

@dsl.pipeline(
    name="yoda-finetuning-pipeline",
    description="Pipeline pour préparer les données et fine-tuner un modèle Phi-3 sur les phrases de Yoda.",
    pipeline_root=constants.PIPELINE_ROOT
)
def model_training_pipeline(
    raw_data_gcs_path: str = "gs://bucket-llm-ops/yoda_sentences.csv"
):
    """
    Définit le workflow de la pipeline.
    """
    # Étape 1 : Traitement des données
    # On appelle notre composant. Il devient une étape (une "tâche") dans la pipeline.
    # Le compilateur KFP gère les OutputPath, nous n'avons pas besoin de les passer.
    data_processing_task = process_yoda_data(
        raw_data_gcs_path=raw_data_gcs_path
    )

    # Note : Dans une pipeline plus complexe, on pourrait utiliser les sorties de cette
    # tâche comme entrées pour la tâche suivante. Par exemple :
    # training_task = train_model(
    #     train_data=data_processing_task.outputs["train_dataset_path"]
    # )