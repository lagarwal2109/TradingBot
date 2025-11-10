#!/bin/bash
# Start the data collector

set -e

echo "Starting Roostoo Data Collector..."

# Enable and start the collector service
sudo systemctl enable roostoo-collector.service
sudo systemctl start roostoo-collector.service

echo "Data collector started successfully!"
echo
echo "The collector will continuously gather minute bar data."
echo "Use './scripts/status.sh' to check status"
echo "Use 'tail -f logs/bot.log' to view live logs"
