#!/bin/bash

# Fetch secrets from Google Secret Manager
python -c "
import os
from google.cloud import secretmanager

secret_name = os.environ['STREAMLIT_SECRET_NAME']
client = secretmanager.SecretManagerServiceClient()
response = client.access_secret_version(name=secret_name)
payload = response.payload.data.decode('UTF-8')

with open('.streamlit/secrets.toml', 'w') as f:
    f.write(payload)
"

# Start Streamlit with configured port
exec streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0