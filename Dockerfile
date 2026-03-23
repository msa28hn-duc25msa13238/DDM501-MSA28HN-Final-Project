FROM python:3.10-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --uid 50000 --gid 0 --create-home --home-dir /home/appuser appuser \
    && mkdir -p /tmp/matplotlib /tmp/.config \
    && chmod -R 0777 /tmp/matplotlib /tmp/.config /home/appuser

ENV PIP_DEFAULT_TIMEOUT=120

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV HOME=/home/appuser
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV XDG_CONFIG_HOME=/tmp/.config
ENV PYTHONWARNINGS=ignore::UserWarning:_distutils_hack

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
