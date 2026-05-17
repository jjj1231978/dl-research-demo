# Phase 0 stub — Hugging Face Space runtime.
# Phase 4 (deployment) verifies and extends this; the Phase 0 commitment is
# only that the file exists and meets the layout-conformance criterion (SC-006).

FROM python:3.11-slim

WORKDIR /app

# Install Python deps first so Docker layer-caches them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# HF Spaces route traffic to port 7860 by default
EXPOSE 7860

# Belt-and-suspenders: also set these in .streamlit/config.toml
ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "streamlit_app.py"]
