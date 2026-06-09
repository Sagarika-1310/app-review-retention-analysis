import tempfile
import traceback
from datetime import datetime

import mlflow


def mlflow_stage(stage_id: str, stage_name: str):
    """
    Decorator that wraps a stage execution inside
    MLflow tracking, timing, status logging,
    traceback artifact logging, and exception propagation.
    """

    def decorator(func):

        def wrapper(*args, **kwargs):

            with mlflow.start_run(
                    run_name=stage_name,
                    nested=True,
            ):

                mlflow.set_tag("stage_id", stage_id)
                mlflow.set_tag("stage_name", stage_name)
                mlflow.set_tag("stage_status", "RUNNING")

                start_time = datetime.now()

                try:

                    result = func(*args, **kwargs)

                    duration = (
                            datetime.now() - start_time
                    ).total_seconds()

                    mlflow.log_metric(
                        "duration_seconds",
                        duration,
                    )

                    mlflow.set_tag(
                        "stage_status",
                        "SUCCESS",
                    )

                    return result

                except Exception as exc:

                    duration = (
                            datetime.now() - start_time
                    ).total_seconds()

                    mlflow.log_metric(
                        "duration_seconds",
                        duration,
                    )

                    mlflow.set_tag(
                        "stage_status",
                        "FAILED",
                    )

                    mlflow.set_tag(
                        "failure_reason",
                        str(exc),
                    )

                    tb = traceback.format_exc()

                    with tempfile.NamedTemporaryFile(
                            mode="w",
                            suffix=".log",
                            delete=False,
                            encoding="utf-8",
                    ) as f:

                        f.write(tb)

                        traceback_file = f.name

                    mlflow.log_artifact(
                        traceback_file,
                        artifact_path="errors",
                    )

                    raise

        return wrapper

    return decorator
