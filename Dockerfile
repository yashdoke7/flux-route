FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy project
COPY . .

# Create results dir
RUN mkdir -p results agent/checkpoints

# Expose HF Spaces port
EXPOSE 7860

# Make start script executable
RUN chmod +x start.sh

CMD ["./start.sh"]
