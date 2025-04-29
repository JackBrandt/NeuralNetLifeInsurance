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

python -c "
import subprocess
import streamlit as st

def start_auto_emailer():
    # Using start_new_session=True to detach the child process from the parent,
    # so it won't be terminated when the Streamlit app stops.
    process = subprocess.Popen(
        ['python', 'auto_emailer.py'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )

start_auto_emailer()
"

# Start Streamlit with configured port
exec streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0