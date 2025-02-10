#!/bin/bash

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:${PWD}"
export FLASK_APP=app.py
export FLASK_ENV=production

# Start Gunicorn with minimal configuration for Render
exec gunicorn \
    --workers=2 \
    --threads=4 \
    --timeout=120 \
    --worker-class=gthread \
    --worker-tmp-dir=/dev/shm \
    --max-requests=200 \
    --max-requests-jitter=50 \
    --log-file=- \
    --access-logfile=- \
    --error-logfile=- \
    --log-level=info \
    --bind=0.0.0.0:$PORT \
    app:app
