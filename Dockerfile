FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy project
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts /app/scripts

# Install
RUN pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir -e /app

EXPOSE 8000

# Render/most platforms provide PORT; default to 8000 locally
CMD ["sh", "-c", "python scripts/fetch_model.py && uvicorn mvgrid.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
