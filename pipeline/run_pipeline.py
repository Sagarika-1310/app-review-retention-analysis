"""
run_pipeline.py
---------------
Centralized orchestrator for the Spotify Retention Risk pipeline.

Responsibilities:
- Environment initialization
- MLflow configuration
- Parent run management
- Pipeline version generation
- Stage orchestration
- Pipeline-level success/failure tracking
"""

import os
import logging
from datetime import datetime

import mlflow
from dotenv import load_dotenv

# Stage imports
from stages import stage_1_scraper
from stages import stage_2_eda
from stages import stage_3_data_processor
from stages import stage_4a_sentiment_evaluation
from stages import stage_4b_sentiment_scoring
from stages import stage_5_risk_scoring
from stages import stage_6a_baseline_embeddings_extraction
from stages import stage_6b_baseline_pain_point_clustering
from stages import stage_7a_incremental_assignment
from stages import stage_7b_discovery_run
from stages import stage_8_validation

# Decorator executor
from pipeline.decorators.mlflow_stage import mlflow_stage

# ──────────────────────────────────────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

os.makedirs(
    os.path.join(PROJECT_ROOT, "logs"),
    exist_ok=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(
                PROJECT_ROOT,
                "logs",
                "pipeline_orchestrator.log",
            ),
            encoding="utf-8",
        ),
    ],
)

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Configuration
# ──────────────────────────────────────────────────────────────────────────────

PIPELINE_CONFIG = {
    "experiment_name": "Spotify_Retention_Risk_Scorer",
    "stages": [
        {
            "id": "stage_1_scraping",
            "name": "Stage_1_Review_Scraping",
            "entrypoint": stage_1_scraper.run,
            "enabled": False,
        },
        {
            "id": "stage_2_eda",
            "name": "Stage_2_EDA",
            "entrypoint": stage_2_eda.run,
            "enabled": True,
        },
        {
            "id": "stage_3_data_processor",
            "name": "Stage_3_Data_Processor",
            "entrypoint": stage_3_data_processor.run,
            "enabled": False,
        },
        {
            "id": "stage_4a_sentiment_evaluation",
            "name": "Stage_4A_Sentiment_Svaluation",
            "entrypoint": stage_4a_sentiment_evaluation.run,
            "enabled": False,
        },
        {
            "id": "stage_4b_sentiment_scoring",
            "name": "Stage_4B_Sentiment_Scoring",
            "entrypoint": stage_4b_sentiment_scoring.run,
            "enabled": False,
        },
        {
            "id": "stage_5_risk_scoring",
            "name": "Stage_5_Risk_Scoring",
            "entrypoint": stage_5_risk_scoring.run,
            "enabled": False,
        },
        {
            # Only run first time to set baseline topics
            "id": "stage_6a_baseline_embeddings_extraction",
            "name": "Stage_6A_Baseline_Embeddings_Extraction",
            "entrypoint": stage_6a_baseline_embeddings_extraction.run,
            "enabled": False,
        },
        {
            # Only run first time to set baseline topics
            "id": "stage_6b_baseline_pain_point_clustering",
            "name": "Stage_6B_Baseline_Pain_Point_Clustering",
            "entrypoint": stage_6b_baseline_pain_point_clustering.run,
            "enabled": False,
        },
        {
            "id": "stage_7a_incremental_assignment",
            "name": "Stage_7A_Incremental_Assignment",
            "entrypoint": stage_7a_incremental_assignment.run,
            "enabled": False,
        },
        {
            "id": "stage_7b_discovery_run",
            "name": "Stage_7B_Discovery_Run",
            "entrypoint": stage_7b_discovery_run.run,
            "enabled": False,
        },
        {
            "id": "stage_8_validation",
            "name": "Stage_8_Validation",
            "entrypoint": stage_8_validation.run,
            "enabled": False,
        },
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# Stage Executor
# ──────────────────────────────────────────────────────────────────────────────

def execute_stage(stage, run_version):
    """
    Dynamically decorates and executes a stage.
    """

    decorated_stage = mlflow_stage(
        stage_id=stage["id"],
        stage_name=stage["name"],
    )(stage["entrypoint"])

    decorated_stage(
        run_version=run_version,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 80)
    log.info("INITIALIZING END-TO-END PIPELINE ORCHESTRATION ENGINE")
    log.info("=" * 80)

    # -------------------------------------------------------------------------
    # Environment Setup
    # -------------------------------------------------------------------------

    env_path = os.path.join(
        PROJECT_ROOT,
        ".env",
    )

    load_dotenv(
        dotenv_path=env_path,
    )

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI"
    )

    if tracking_uri:
        mlflow.set_tracking_uri(
            tracking_uri,
        )

        log.info(
            f"MLflow tracking engine bound to: "
            f"{tracking_uri}"
        )

    mlflow.set_experiment(
        PIPELINE_CONFIG["experiment_name"]
    )

    # -------------------------------------------------------------------------
    # Global Versioning
    # -------------------------------------------------------------------------

    run_version = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    parent_run_name = (
        f"End_to_End_Retention_Pipeline_"
        f"{run_version}"
    )

    start_time = datetime.now()

    # -------------------------------------------------------------------------
    # Parent MLflow Run
    # -------------------------------------------------------------------------

    with mlflow.start_run(
            run_name=parent_run_name
    ):

        mlflow.set_tag(
            "pipeline_status",
            "RUNNING",
        )

        mlflow.set_tag(
            "pipeline_version",
            run_version,
        )

        mlflow.log_param(
            "global_pipeline_version",
            run_version,
        )

        mlflow.log_param(
            "execution_timestamp",
            start_time.isoformat(),
        )

        try:

            log.info(
                f"Successfully spawned "
                f"Parent Pipeline Run: "
                f"{parent_run_name}"
            )

            # -------------------------------------------------------------
            # Execute Enabled Stages
            # -------------------------------------------------------------

            for stage in PIPELINE_CONFIG["stages"]:

                if not stage["enabled"]:
                    log.info(
                        f"Stage [{stage['name']}] "
                        f"is DISABLED. Skipping."
                    )

                    continue

                log.info("\n" + "─" * 80)
                log.info(
                    f"LAUNCHING STAGE: "
                    f"{stage['name']}"
                )
                log.info("─" * 80)

                execute_stage(
                    stage=stage,
                    run_version=run_version,
                )

            # -------------------------------------------------------------
            # Success
            # -------------------------------------------------------------

            elapsed = (
                    datetime.now() - start_time
            ).total_seconds()

            mlflow.log_metric(
                "pipeline_duration_seconds",
                elapsed,
            )

            mlflow.set_tag(
                "pipeline_status",
                "SUCCESS",
            )

            log.info("\n" + "=" * 80)
            log.info(
                "END-TO-END PIPELINE EXECUTION COMPLETE"
            )
            log.info(
                f"Total Duration: "
                f"{elapsed:.2f} seconds"
            )
            log.info("=" * 80)

        except Exception as exc:

            elapsed = (
                    datetime.now() - start_time
            ).total_seconds()

            mlflow.log_metric(
                "pipeline_duration_seconds",
                elapsed,
            )

            mlflow.set_tag(
                "pipeline_status",
                "FAILED",
            )

            mlflow.set_tag(
                "failure_reason",
                str(exc),
            )

            log.exception(
                "PIPELINE EXECUTION FAILED"
            )

            raise


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
