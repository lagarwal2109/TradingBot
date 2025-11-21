#!/bin/bash
# Check status of bot services

echo "=== Roostoo Bot Status ==="
echo

echo "Trading Bot Timer:"
sudo systemctl status roostoo-bot.timer --no-pager | grep -E "Active:|Trigger:" || echo "Not found"
echo

echo "Trading Bot Service:"
sudo systemctl status roostoo-bot.service --no-pager | head -n 5 || echo "Not found"
echo

echo "Data Collector:"
sudo systemctl status roostoo-collector.service --no-pager | head -n 5 || echo "Not found"
echo

echo "=== Recent Log Entries ==="
if [ -f "logs/bot.log" ]; then
    echo "Last 10 bot log entries:"
    tail -n 10 logs/bot.log 2>/dev/null || echo "No logs yet"
fi
