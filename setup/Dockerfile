# Stage 1: Build React frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/index.html frontend/vite.config.js ./
COPY frontend/public/ ./public/
COPY frontend/src/ ./src/
RUN npm run build

# Stage 2: Python API + serve built frontend
FROM python:3.11-slim
WORKDIR /app

COPY setup/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY setup/config.toml ./
COPY src/ src/
COPY scripts/ scripts/
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 8060
CMD ["python", "scripts/run_server.py"]
