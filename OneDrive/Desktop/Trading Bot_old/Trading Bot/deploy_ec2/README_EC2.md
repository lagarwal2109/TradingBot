EC2 Deployment - Volatility Expansion Bot

Prereqs
- Ubuntu 22.04 LTS (recommended)
- Python 3.10+
- A user with sudo
- Your Roostoo API key/secret

Quick Start (copy/paste)
1) Update system and install Python:
   sudo apt update && sudo apt -y upgrade
   sudo apt -y install python3-pip python3-venv git

2) Create app directory and copy repo:
   mkdir -p ~/app && cd ~/app
   # Copy your project to this path (scp/rsync) or git clone your repo
   # Then ensure this deploy folder exists at: ~/app/TradingBot/deploy_ec2

3) Create .env with your keys (replace values):
   cd ~/app/TradingBot/deploy_ec2
   cp .env.example .env
   nano .env
   # Set ROOSTOO_API_KEY and ROOSTOO_API_SECRET

4) One-line setup (venv + deps + service install):
   sudo bash setup.sh

5) Start the bot:
   sudo systemctl start roostoo-bot
   sudo systemctl status roostoo-bot

6) Follow logs:
   journalctl -u roostoo-bot -f
   # Bot-specific log:
   tail -f ~/app/TradingBot/logs/volatility_expansion_live.log

Stop/Start/Restart
- Stop: sudo systemctl stop roostoo-bot
- Start: sudo systemctl start roostoo-bot
- Restart: sudo systemctl restart roostoo-bot

Pre-populate market data (optional, recommended)
- If you already have CSV minute data under TradingBot/data, copy it to the server to start trading from minute 1.
- If not, the bot will still run; it will warm up as minute bars accumulate.

What this installer does
- Creates a Python venv in ~/app/venv
- Installs Python dependencies from requirements.txt
- Creates a systemd service to keep the bot running and auto-restart
- Uses your .env (ROOSTOO credentials) to authenticate

Files of interest
- deploy_ec2/.env.example   # template for secrets
- deploy_ec2/requirements.txt
- deploy_ec2/setup.sh       # installs venv and service
- deploy_ec2/roostoo-bot.service  # systemd unit
- run_volatility.py         # live launcher

Troubleshooting
- Status: sudo systemctl status roostoo-bot
- Logs: journalctl -u roostoo-bot -f
- App logs: tail -f ~/app/TradingBot/logs/volatility_expansion_live.log
- Ensure .env keys are correct; base URL defaults to https://mock-api.roostoo.com, set ROOSTOO_BASE_URL if needed.


