# Agent container. Deliberately contains no game-server code and no scenario data.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates bash \
    && rm -rf /var/lib/apt/lists/*
# Used by the coded reference in solution/ (an MCP client); also there for an agent allowed to code.
RUN pip install --no-cache-dir mcp==2.3.0 numpy==1.26.4
# The evaluated agent is a player: the game's MCP tools only, no shell, no file or web access.
ENV OPENCODE_CONFIG_CONTENT='{"tools":{"bash":false,"edit":false,"write":false,"patch":false,"multiedit":false,"read":false,"grep":false,"glob":false,"list":false,"webfetch":false,"websearch":false,"task":false,"skill":false,"lsp":false,"codesearch":false,"google_search":false},"permission":{"bash":"deny","edit":"deny","webfetch":"deny","websearch":"deny"}}'
WORKDIR /app
