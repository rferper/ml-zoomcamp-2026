# the base image: Python 3.12 on a minimal Debian. The slim variant is smaller, which makes the image faster to build and to download.
# The version should match your local Python (the one in .python-version), so the container runs the same Python you tested with
FROM python:3.12-slim

# install uv inside the image. Instead of "pip install", we copy the uv binary directly from uv's official image.
# COPY --from=<image> takes files out of another image, so we get uv without installing anything with pip.
# uv is a single executable, so this is all it needs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# tell uv to use the Python that already comes with the base image, instead of downloading its own copy
ENV UV_PYTHON_DOWNLOADS=never

# create the /app directory and make it the working directory of the following instructions
WORKDIR /app

# copy the dependency files into the image: pyproject.toml plays the role of the Pipfile, uv.lock the role of Pipfile.lock,
# and .python-version records the Python version (like the [requires] section of the Pipfile).
# We copy them BEFORE the code on purpose: Docker caches each step (layer), so if only predict.py changes,
# Docker reuses the cached installation below instead of reinstalling all the packages
COPY ["pyproject.toml", "uv.lock", ".python-version", "./"]

# install exactly what the lock file says. Unlike pipenv's --system, uv installs the packages into a virtual environment at /app/.venv.
# The container is already isolated, but this is the pattern uv recommends and it keeps the packages in one known place.
# --locked makes uv fail if the lock file is not up to date with pyproject.toml (the equivalent of pipenv's --deploy),
# --no-dev skips development-only packages like jupyter, which the service doesn't need,
# and --no-install-project installs only the dependencies, not our own project as a package (its source code isn't copied yet)
RUN uv sync --locked --no-dev --no-install-project

# put the virtual environment's bin folder first in the PATH, so that commands like "python" and "gunicorn" are found in /app/.venv.
# This is the same thing that activating a virtual environment does on your machine, but done once for the whole container
ENV PATH="/app/.venv/bin:$PATH"

# copy the service code and the model into the image. The paths are relative to the build context (the project root),
# which is why they include src/ml_zoomcamp_2026. Inside the image both files land directly in /app
COPY ["src/ml_zoomcamp_2026/predict.py", "src/ml_zoomcamp_2026/model_C=1.0.bin", "./"]

# document that the container listens on port 9696. The container's network is isolated, so the port has to be published when starting the container
EXPOSE 9696

# the command that runs when the container starts: our production server serving the Flask app.
# Inside Docker we're on Linux, so gunicorn works here even though it doesn't on Windows.
# Without an ENTRYPOINT, the container would run the base image's default command (a Python shell) instead of our service
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
