FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt /app/
RUN python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu && python -m pip install -r requirements.txt && python -m pip check && python -m pip freeze > /app/environment.lock.txt
COPY adhd /app/adhd
COPY configs /app/configs
COPY sql.sql /app/
COPY tests /app/tests
ARG PIPELINE_UID=1000
ARG PIPELINE_GID=1000
RUN groupadd --gid "${PIPELINE_GID}" pipeline && useradd --uid "${PIPELINE_UID}" --gid pipeline --create-home pipeline
USER pipeline
CMD ["python","-m","adhd.cli","--help"]
