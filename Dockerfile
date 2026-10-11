FROM python:3.12-slim

# AWS Lambda Web Adapter: allows FastAPI to run unchanged on AWS Lambda, ECS, or locally
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.9.0 /lambda-adapter /opt/extensions/lambda-adapter
ENV PORT=8000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Copy dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy source code and the self-contained production model bundle
COPY src/ ./src/
COPY models/production/ ./models/production/

EXPOSE 8000

# 3. Bind to 0.0.0.0 for external reachability
CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
