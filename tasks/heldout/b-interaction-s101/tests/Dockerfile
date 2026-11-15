# Verifier image (separate from the agent's container, no network).
FROM python:3.12-slim
COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY . /tests/
ENV PYTHONPATH=/tests PYTHONWARNINGS=ignore
