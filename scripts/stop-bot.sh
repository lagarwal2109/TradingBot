#!/bin/bash
# Stop the trading bot

set -e

echo "Stopping Roostoo Trading Bot..."

# Stop and disable the timer
sudo systemctl stop roostoo-bot.timer
sudo systemctl disable roostoo-bot.timer

# Also stop the service if running
sudo systemctl stop roostoo-bot.service || true

echo "Trading bot stopped successfully!"
