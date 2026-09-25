# syntax=docker/dockerfile:1

# --- Stage 1: build the Svelte frontend -------------------------------------
FROM --platform=$BUILDPLATFORM node:24-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# Vite's outDir (../parallax_maker/static/app, see frontend/vite.config.ts)
# resolves relative to /app/frontend, i.e. /app/parallax_maker/static/app.
RUN npm run build

# --- Stage 2: the Python application, with the built frontend included -----
# Built for the target platform (only the Node stage above is platform-neutral).
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
# The frontend is packaged via [tool.setuptools.package-data]
# (parallax_maker/static/app/**/*); it must exist before `pip install .` so
# setuptools picks it up.
COPY --from=frontend-build /app/parallax_maker/static/app /app/parallax_maker/static/app
RUN pip install --no-cache-dir .

EXPOSE 8050

# The .cache directory will be optionally bound to /root/.cache/
VOLUME ["/root/.cache/"]

# The working directory can be mounted to /app/workdir
VOLUME ["/app/workdir"]

WORKDIR /app/workdir

CMD ["parallax-maker", "--host", "0.0.0.0", "--port", "8050"]
