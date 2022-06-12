import prefect
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule

DeploymentSpec(
    name="cron-schedule-deployment",
    flow_location="D:\\Files\\mlops\\mlops-zoomcamp\\mlops-zoomcamp\\03-orchestration\\homework-11-06-2022.py",
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="Asia/Kolkata"),
)
