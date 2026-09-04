# Roadstead, containerised.
#
# 🚨 STOP-GRACE-PERIOD IS LOAD-BEARING. SIGTERM runs a bounded drain that
# persists DRR budgets and completion rows, and uvicorn's connection budget and
# the app's drain budget are SERIAL, not nested (docs/ledger.md). `docker stop`
# sends SIGTERM and hard-kills after **10s by default**, which truncates the
# drain in every non-idle case and silently loses state. Run it with
# `--stop-timeout 108`, or `stop_grace_period: 108s` in compose. The STOPSIGNAL
# and the label below make the requirement discoverable from the image itself
# rather than only from a document.
#
# 🚨 The number went UP from 90 on 2026-09-01, and that is the fix rather than a
# regression. 90 came from a MEASURED worst case (~78s) at a time when the tail
# after the drain — releasing GPU leases, closing HTTP pools — had no bound at
# all: a wedged dispatcher could hang shutdown indefinitely and 90 would not
# have been enough. Every phase is bounded now, and 108 is the first stop-grace
# that is a ceiling rather than an observation. It is computed in ONE place,
# `service.RECOMMENDED_STOP_GRACE_S`, and `tests/test_shutdown_budget.py` fails
# if this label disagrees with it.
FROM python:3.11-slim-bookworm

LABEL org.opencontainers.image.title="roadstead" \
      org.opencontainers.image.description="Capacity-aware admission control for self-hosted LLM inference fleets" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.roadstead.required-stop-grace-period-seconds="108"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 🚨 The durable event log, OFF the container's ephemeral writable layer.
# Without this the default data dir is `/tmp/agents/llmproxy` — right for a
# developer running the module directly, wrong for the artifact that ships.
# Measured in a real container 2026-09-02: `queue.db` sat on the writable layer
# while the mounted volume held only the admin overlay, so every rebuild
# silently reset the DRR balances, the day's spend and the endpoint drain state
# — the exact rows the bounded SIGTERM drain and the 108s stop-grace above exist
# to flush. The whole shutdown budget was protecting a file the next
# `up --build` deleted.
#
# Its own ENV, not appended to the block above: a comment inside a line
# continuation is not portable across Dockerfile parsers, and this one has to
# carry its reason.
ENV ROADSTEAD_DATA_DIR=/var/lib/roadstead

# Declared so `docker run` without `-v` still gets an anonymous volume rather
# than the writable layer, and so `docker inspect` names the path an operator
# has to mount. 🚨 A VOLUME is not a substitute for mounting a real one — an
# anonymous volume survives a restart but not a `down -v`, and is orphaned by a
# recreate. It moves the default from "certainly lost" to "not silently lost".
VOLUME ["/var/lib/roadstead"]

WORKDIR /app

# Dependency layer first, so a source edit does not re-resolve the world.
COPY pyproject.toml README.md ./
COPY roadstead/__init__.py roadstead/
RUN pip install --no-cache-dir -e . && rm -rf /root/.cache

COPY roadstead/ roadstead/

# The default; `docker stop` sends it anyway. Stated so the image documents it.
STOPSIGNAL SIGTERM

# 🚨 STAYS ROOT — a considered trade-off, not an oversight. The only real
# deployment of this image today bind-mounts a ZFS path (`ROADSTEAD_DATA_DIR`)
# that is owned by root on the host, inside an unprivileged LXC — so the host
# side of that mount cannot be re-owned to a container UID without the
# operator's cooperation, and a bind mount's permissions come from the HOST,
# not from anything `chown`ed at image-build time. Flipping the default user
# here would not make the image "more non-root" for that deployment; it would
# just make `/var/lib/roadstead` unwritable the next time the image is
# rebuilt, silently, on a path nobody is watching for a permissions error.
#
# The two ways to actually get non-root without that landmine both need action
# OUTSIDE this file: (a) `chown` the host-side data directory to a fixed
# container UID before the next `up`, or (b) `chmod 0777` it — no sticky bit,
# since it holds one app's data, not several tenants' — and either is a
# decision for whoever owns that host path, not a default this image can pick
# for them. Until one of those happens, HEALTHCHECK below is the hardening
# that ships unconditionally; non-root is tracked as needing the deployment's
# cooperation, not blocked on code here.
EXPOSE 42161

# Liveness, not readiness: `/health` fails OPEN on a dead BACKEND by design
# (alert-don't-kill, see `http_handlers.handle_health`) and only 503s when the
# scheduler loop itself has died — exactly the "is the process still alive"
# question a container orchestrator should be asking. `/readyz` fails CLOSED
# on backend/circuit-breaker state instead, which is right for routing and
# wrong here: it would flap the container unhealthy over a backend blip that
# the code is deliberately built not to restart the proxy for. Plain
# `urllib.request` because the base image carries no `curl`.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '42161') + '/health', timeout=3)"

ENTRYPOINT ["python", "-m", "roadstead"]
