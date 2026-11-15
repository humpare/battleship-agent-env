FROM python:3.12-slim
RUN pip install --no-cache-dir mcp==2.3.0 numpy==1.26.4
COPY battleship /opt/app/battleship
COPY scenario.json /opt/app/scenario.json
ENV PYTHONPATH=/opt/app PYTHONUNBUFFERED=1 BATTLESHIP_N=5
RUN mkdir -p /data
CMD ["python", "-m", "battleship.duel_server", "--scenario", "/opt/app/scenario.json", "--log", "/data/episode.jsonl"]
