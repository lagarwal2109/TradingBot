#!/bin/bash
# Stop the data collector

set -e

echo "Stopping Roostoo Data Collector..."

# Stop and disable the collector service
sudo systemctl stop roostoo-collector.service
sudo systemctl disable roostoo-collector.service

echo "Data collector stopped successfully!"
