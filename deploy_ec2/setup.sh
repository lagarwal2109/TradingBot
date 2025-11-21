#!/usr/bin/env bash
set -euo pipefail

# Paths
APP_ROOT="$HOME/app"
PROJECT_DIR="$APP_ROOT/TradingBot"
VENV_DIR="$APP_ROOT/venv"
SERVICE_NAME="roostoo-bot"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "Creating app directories..."
mkdir -p "$APP_ROOT"
mkdir -p "$PROJECT_DIR/logs"

echo "Creating Python venv..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r "$PROJECT_DIR/deploy_ec2/requirements.txt"

echo "Writing systemd service..."
sudo tee "$SERVICE_FILE" >/dev/null <<EOF
[Unit]
Description=Roostoo Volatility Expansion Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${PROJECT_DIR}
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONIOENCODING=utf-8
EnvironmentFile=${PROJECT_DIR}/deploy_ec2/.env
ExecStart=${VENV_DIR}/bin/python ${PROJECT_DIR}/run_volatility.py
Restart=always
RestartSec=5
StandardOutput=append:${PROJECT_DIR}/logs/service.out.log
StandardError=append:${PROJECT_DIR}/logs/service.err.log

[Install]
WantedBy=multi-user.target
EOF

echo "Reloading systemd..."
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"

echo "Setup complete. Start with: sudo systemctl start ${SERVICE_NAME}"

