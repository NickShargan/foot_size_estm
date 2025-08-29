# Use a slim Python base
FROM python:3.11-slim

# Install system dependencies (needed for OpenCV & Matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# # 1) Install CPU-only torch/torchvision explicitly
# RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
#     torch==2.4.0 torchvision==0.19.0

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app ./app

# Environment variables
ENV PORT=8080 \
    PYTHONUNBUFFERED=1

# (Optional) place your model weights at build time:
# COPY sam2.1_b.pt /app/sam2.1_b.pt
# ENV SAM_WEIGHTS=/app/sam2.1_b.pt
#COPY sam2.1_b.pt /app/sam2.1_t.pt
#ENV SAM_WEIGHTS=/app/sam2.1_t.pt

EXPOSE 8080
# CMD ["python", "-m", "app.main"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
