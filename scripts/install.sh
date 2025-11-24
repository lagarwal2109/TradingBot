#!/bin/bash
# Installation script for Roostoo Trading Bot on Ubuntu

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the absolute path of the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo -e "${GREEN}Installing Roostoo Trading Bot...${NC}"
echo "Project directory: $PROJECT_DIR"

# Check if running on Ubuntu/Debian
if ! command -v apt-get &> /dev/null; then
    echo -e "${RED}This script is designed for Ubuntu/Debian systems${NC}"
    exit 1
fi

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo -e "${RED}Please do not run this script as root${NC}"
   exit 1
fi

# Update system packages
echo -e "${YELLOW}Updating system packages...${NC}"
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip git

# Create virtual environment
echo -e "${YELLOW}Creating Python virtual environment...${NC}"
cd "$PROJECT_DIR"
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Create necessary directories
echo -e "${YELLOW}Creating data directories...${NC}"
mkdir -p data logs figures

# Check for .env file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cat > .env << EOF
# Roostoo API Configuration
ROOSTOO_API_KEY=your_api_key_here
ROOSTOO_API_SECRET=your_api_secret_here
ROOSTOO_BASE_URL=https://api.roostoo.com
EOF
    echo -e "${RED}Please edit .env file and add your API credentials${NC}"
fi

# Make scripts executable
chmod +x run.py backtest.py
chmod +x scripts/*.sh

# Create systemd service files
echo -e "${YELLOW}Creating systemd service files...${NC}"
sudo cp scripts/systemd/roostoo-bot.service /etc/systemd/system/
sudo cp scripts/systemd/roostoo-bot.timer /etc/systemd/system/
sudo cp scripts/systemd/roostoo-collector.service /etc/systemd/system/

# Update service files with correct paths
sudo sed -i "s|/path/to/roostoo-sharpe-bot|$PROJECT_DIR|g" /etc/systemd/system/roostoo-bot.service
sudo sed -i "s|/path/to/roostoo-sharpe-bot|$PROJECT_DIR|g" /etc/systemd/system/roostoo-collector.service

# Create bot user if needed (optional - for better security)
if ! id -u roostoo-bot &>/dev/null; then
    echo -e "${YELLOW}Creating roostoo-bot user...${NC}"
    sudo useradd -r -s /bin/false roostoo-bot || true
fi

# Set ownership
sudo chown -R $USER:$USER "$PROJECT_DIR"

# Reload systemd
echo -e "${YELLOW}Reloading systemd...${NC}"
sudo systemctl daemon-reload

echo -e "${GREEN}Installation complete!${NC}"
echo
echo "Next steps:"
echo "1. Edit .env file with your API credentials"
echo "2. Run collector first to gather data: ./scripts/start-collector.sh"
echo "3. After collecting data, run backtest: python backtest.py"
echo "4. Start the trading bot: ./scripts/start-bot.sh"
echo
echo "Available commands:"
echo "  ./scripts/start-bot.sh       - Start trading bot"
echo "  ./scripts/stop-bot.sh        - Stop trading bot"
echo "  ./scripts/start-collector.sh - Start data collector"
echo "  ./scripts/stop-collector.sh  - Stop data collector"
echo "  ./scripts/status.sh          - Check bot status"
echo "  ./scripts/logs.sh            - View bot logs"
