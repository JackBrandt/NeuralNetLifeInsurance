FROM mcr.microsoft.com/devcontainers/anaconda:1-3

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./

# Install dependencies including Secret Manager client
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install google-cloud-secret-manager

# Install Google Cloud SDK
RUN apt-get update && apt-get install -y curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" >> /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-sdk

# Copy application files
COPY . .

# Create .streamlit directory and copy entrypoint
RUN mkdir -p .streamlit
COPY .devcontainer/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Conda environment setup
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

ENV PORT=8501
ENTRYPOINT ["/app/entrypoint.sh"]
