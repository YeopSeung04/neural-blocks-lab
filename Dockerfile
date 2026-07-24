FROM node:22-alpine AS frontend-dependencies
WORKDIR /build
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=frontend-dependencies /build/node_modules ./node_modules

EXPOSE 8080
CMD ["python", "run_server.py", "--bind", "0.0.0.0", "--port", "8080"]
