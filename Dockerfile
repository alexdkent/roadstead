# Roadstead, containerised.
#
# 🚨 STOP-GRACE-PERIOD IS LOAD-BEARING. SIGTERM runs a bounded drain that
# persists DRR budgets and completion rows, and uvicorn's connection budget and
# the app's drain budget are SERIAL, not nested (docs/ledger.md). Worst case
# measured at ~78s. `docker stop` sends SIGTERM and hard-kills after **10s by
# default**, which truncates the drain in every non-idle case and silently loses
# state. Run it with `--stop-timeout 90`, or `stop_grace_period: 90s` in compose.
# The STOPSIGNAL and the label below make the requirement discoverable from the
# image itself rather than only from a document.
FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="roadstead" \
      org.opencontainers.image.description="Capacity-aware admission control for self-hosted LLM inference fleets" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.roadstead.required-stop-grace-period-seconds="90"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependency layer first, so a source edit does not re-resolve the world.
COPY pyproject.toml README.md ./
COPY roadstead/__init__.py roadstead/
RUN pip install --no-cache-dir -e . && rm -rf /root/.cache

COPY roadstead/ roadstead/

# The default; `docker stop` sends it anyway. Stated so the image documents it.
STOPSIGNAL SIGTERM

EXPOSE 42161
ENTRYPOINT ["python", "-m", "roadstead"]
