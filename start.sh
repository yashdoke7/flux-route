#!/bin/bash
set -e

echo "Starting FluxRoute server on port 7860..."
exec uvicorn server:app --host 0.0.0.0 --port 7860 --log-level info
