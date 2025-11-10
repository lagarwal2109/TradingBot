#!/bin/bash
# Check available data for backtesting

echo "=== Data Availability Check ==="
echo

DATA_DIR="data"

if [ ! -d "$DATA_DIR" ]; then
    echo "Data directory not found. Run the collector first."
    exit 1
fi

# Count CSV files
CSV_COUNT=$(find "$DATA_DIR" -name "*.csv" -not -name "state.json" | wc -l)
echo "Trading pairs with data: $CSV_COUNT"
echo

if [ $CSV_COUNT -gt 0 ]; then
    echo "Data files:"
    for file in "$DATA_DIR"/*.csv; do
        if [ -f "$file" ] && [ "$(basename "$file")" != "state.json" ]; then
            lines=$(wc -l < "$file")
            pair=$(basename "$file" .csv | tr '_' '/')
            echo "  $pair: $((lines - 1)) minute bars"
        fi
    done
else
    echo "No data files found. Please run the collector:"
    echo "  python run.py --mode collect"
    echo "or"
    echo "  ./scripts/start-collector.sh"
fi

echo
echo "Minimum data required: 241 minute bars per pair for backtesting"
