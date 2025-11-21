#!/bin/bash
# View bot logs

if [ "$1" == "-f" ] || [ "$1" == "--follow" ]; then
    # Follow mode
    echo "Following bot logs (Ctrl+C to stop)..."
    tail -f logs/bot.log
else
    # Show recent logs
    echo "=== Recent Bot Logs ==="
    if [ -f "logs/bot.log" ]; then
        tail -n 50 logs/bot.log
    else
        echo "No bot logs found"
    fi
    
    echo
    echo "=== Recent System Logs ==="
    sudo journalctl -u roostoo-bot.service -u roostoo-bot.timer -u roostoo-collector.service --no-pager -n 20
    
    echo
    echo "Tip: Use './scripts/logs.sh -f' to follow logs in real-time"
fi
