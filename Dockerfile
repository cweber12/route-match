################################################################################
# Stage 1 — builder: install Python deps & build wheels on Alpine
################################################################################

FROM python:3.11-alpine AS builder
# Install build dependencies
RUN apk add --no-cache \
    build-base \
    gcc \
    musl-dev \
    libffi-dev \
    openssl-dev \
    ffmpeg-dev \
    jpeg-dev \
    zlib-dev \
    tiff-dev \
    libpng-dev \
    openblas-dev \
    freetype-dev \
    linux-headers \
    pkgconfig

WORKDIR /tmp/app
# Copy only requirements.txt first to leverage Docker cache for dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt
# Now copy the rest of the application code (this layer will always rebuild if code changes)
COPY . .

################################################################################
# Stage 2 — runtime: minimal Alpine with only runtime deps
################################################################################
FROM python:3.11-alpine

RUN apk add --no-cache \
    ffmpeg \
    libjpeg-turbo \
    zlib \
    tiff \
    libpng \
    openblas \
    freetype \
    libwebp \
    libwebp-dev

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AWS_DEFAULT_REGION=us-east-2

# Copy installed Python packages from builder
COPY --from=builder /usr/local /usr/local

WORKDIR /app
# Copy only the application code from builder (this will always be fresh)
COPY --from=builder /tmp/app /app

RUN mkdir -p /app/temp_uploads

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
