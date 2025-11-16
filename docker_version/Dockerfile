# 1. Use official Python base image

FROM python:3.12-slim

# 2. set Working Directory

WORKDIR /app

# 3. copy files

COPY . /app

# 4. Install Dependencies

RUN pip install --no-cache-dir -r requirements.txt

# 5. Expose port

EXPOSE 8200

# 6. Command to run server

CMD ["uvicorn", "model_api:app", "--host", "0.0.0.0", "--port", "8200"]
