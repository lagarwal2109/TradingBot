#!/usr/bin/env python3
"""Test Roostoo API connectivity and endpoints."""

import requests
import time
import hmac
import hashlib
from urllib.parse import urlencode
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ROOSTOO_API_KEY", "test_key")
API_SECRET = os.getenv("ROOSTOO_API_SECRET", "test_secret")
BASE_URL = os.getenv("ROOSTOO_BASE_URL", "https://mock-api.roostoo.com")

print("ROOSTOO API CONNECTIVITY TEST")
print("=" * 70)
print(f"Base URL: {BASE_URL}")
print(f"API Key: {API_KEY[:20] if API_KEY != 'test_key' else 'NOT CONFIGURED'}...")
print()

# Test 1: Check if server is reachable at all
print("1. Testing base URL connectivity...")
try:
    response = requests.get(BASE_URL, timeout=5)
    print(f"   Status: {response.status_code}")
    print(f"   Server reachable: YES")
except requests.exceptions.Timeout:
    print("   ERROR: Timeout - server not responding")
except Exception as e:
    print(f"   ERROR: {str(e)[:100]}")

# Test 2: Public endpoint (should work without auth)
print("\n2. Testing public endpoint /v3/serverTime...")
try:
    response = requests.get(f"{BASE_URL}/v3/serverTime", timeout=10)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Response: {response.json()}")
    else:
        print(f"   Error: {response.text[:200]}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 3: Exchange info (public)
print("\n3. Testing public endpoint /v3/exchangeInfo...")
try:
    response = requests.get(f"{BASE_URL}/v3/exchangeInfo", timeout=10)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Trading pairs available: {len(data.get('TradePairs', {}))}")
        # Show first few pairs
        pairs = list(data.get('TradePairs', {}).keys())[:5]
        print(f"   Sample pairs: {pairs}")
    else:
        print(f"   Error: {response.text[:200]}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 4: Ticker endpoint (needs timestamp)
print("\n4. Testing ticker endpoint /v3/ticker...")
try:
    params = {'timestamp': str(int(time.time() * 1000))}
    response = requests.get(f"{BASE_URL}/v3/ticker", params=params, timeout=10)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        if data.get("Success"):
            tickers = data.get("Data", {})
            print(f"   Tickers available: {len(tickers)}")
            # Show first ticker
            if tickers:
                first_pair = list(tickers.keys())[0]
                print(f"   Sample: {first_pair} = ${tickers[first_pair]['LastPrice']}")
    else:
        print(f"   Error: {response.text[:200]}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 5: Signed endpoint (requires valid credentials)
if API_KEY != "test_key" and API_SECRET != "test_secret":
    print("\n5. Testing signed endpoint /v3/balance...")
    try:
        timestamp = str(int(time.time() * 1000))
        params = {"timestamp": timestamp}
        
        # Create signature (Roostoo v3 way)
        sorted_keys = sorted(params.keys())
        total_params = "&".join(f"{k}={params[k]}" for k in sorted_keys)
        
        signature = hmac.new(
            API_SECRET.encode("utf-8"),
            total_params.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            "RST-API-KEY": API_KEY,
            "MSG-SIGNATURE": signature
        }
        
        response = requests.get(
            f"{BASE_URL}/v3/balance",
            params=params,
            headers=headers,
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            if result.get("Success"):
                print(f"   SUCCESS: Connected to account!")
                wallet = result.get("Wallet", {})
                print(f"   Assets in wallet: {list(wallet.keys())}")
            else:
                print(f"   API Error: {result.get('ErrMsg')}")
        else:
            print(f"   HTTP Error: {response.text[:200]}")
    except Exception as e:
        print(f"   ERROR: {e}")
else:
    print("\n5. Skipping signed endpoint test (no credentials configured)")

print("\n" + "=" * 70)
print("\nDIAGNOSIS:")
print("-" * 70)
print("522 Error = Origin server not responding")
print("\nPossible causes:")
print("1. Roostoo API server is down/overloaded")
print("2. Competition hasn't started (API goes live at start time)")
print("3. Your API credentials aren't activated yet")
print("4. Wrong API base URL")
print("\nRecommended actions:")
print("- Check with hackathon organizers about API status")
print("- Verify competition start time")
print("- Confirm your API credentials are activated")
print("- Try again in a few minutes")
