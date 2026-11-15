# Simulator sidecar: owns the episode, the scenario (including hidden events) and the action log.
FROM python:3.12-slim
COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY dispatch /opt/app/dispatch
COPY scenario.json /opt/app/scenario.json
ENV PYTHONPATH=/opt/app PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
RUN mkdir -p /data
CMD ["python", "-m", "dispatch.server", "--scenario", "/opt/app/scenario.json", "--log", "/data/episode.jsonl"]
