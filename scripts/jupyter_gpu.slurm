#!/bin/bash
#SBATCH --job-name=jupyter-gpu
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=jupyter-gpu-%j.log

set -euo pipefail

# 1. Environment Setup
module load lang/Python/3.11.3-GCCcore-12.3.0

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi
source .venv/bin/activate

# Install Jupyter if not present
if ! command -v jupyter >/dev/null 2>&1; then
    pip install jupyterlab torch
fi

# 2. Network Info
PORT=8888
HOSTNAME=$(hostname)
IP_ADDR=$(hostname -I | awk '{print $1}')

echo "------------------------------------------------------------------"
echo "Jupyter Lab is starting on node: $HOSTNAME"
echo "IP Address: $IP_ADDR"
echo "Port: $PORT"
echo "------------------------------------------------------------------"
echo "To connect from your local machine, run this in a new terminal:"
echo "ssh -L ${PORT}:${HOSTNAME}:${PORT} koa"
echo "------------------------------------------------------------------"

# 3. Launch Jupyter
jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT
