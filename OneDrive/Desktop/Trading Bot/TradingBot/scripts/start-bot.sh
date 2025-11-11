#!/bin/bash
# Start the trading bot

set -e

echo "Starting Roostoo Trading Bot..."

# Enable and start the timer (runs every minute)
sudo systemctl enable roostoo-bot.timer
sudo systemctl start roostoo-bot.timer

# Also enable the service so it starts on boot
sudo systemctl enable roostoo-bot.service

echo "Trading bot started successfully!"
echo "The bot will run every minute."
echo
echo "Use './scripts/status.sh' to check status"
echo "Use './scripts/logs.sh' to view logs"
