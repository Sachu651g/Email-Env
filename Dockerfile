FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Runtime-injectable environment variables
ENV OPENAI_API_KEY=""
ENV MODEL_NAME="gpt-4o-mini"

CMD ["python", "inference.py"]
