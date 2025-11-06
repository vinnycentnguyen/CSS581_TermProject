import asyncio
import websockets
import json
import pandas as pd
import time

async def fetch_volatility_data(start_ts, end_ts, filename="btc_volatility_index.csv"):
    url = "wss://test.deribit.com/ws/api/v2"
    all_data = []

    resolution_ms = 3600 * 1000  # 1 hour
    max_points = 1000            # Deribit’s limit per request

    async with websockets.connect(url) as ws:
        current_start = start_ts

        while current_start < end_ts:
            current_end = min(current_start + (max_points * resolution_ms), end_ts)

            msg = {
                "id": int(time.time() * 1000),
                "jsonrpc": "2.0",
                "method": "public/get_volatility_index_data",
                "params": {
                    "currency": "BTC",
                    "start_timestamp": current_start,
                    "end_timestamp": current_end,
                    "resolution": 3600
                }
            }

            await ws.send(json.dumps(msg))
            response = await ws.recv()
            resp = json.loads(response)

            if "result" not in resp or "data" not in resp["result"]:
                print("No more data or reached end of range.")
                break

            candles = resp["result"]["data"]
            if not candles:
                break

            all_data.extend(candles)
            last_ts = candles[-1][0]
            print(f"Fetched {len(candles)} candles up to {pd.to_datetime(last_ts, unit='ms')}")

            current_start = last_ts + resolution_ms
            await asyncio.sleep(0.1)

    # Convert all data to DataFrame
    df = pd.DataFrame(all_data, columns=["timestamp_ms", "open", "high", "low", "close"])
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    df = df[["datetime", "open", "high", "low", "close"]].sort_values("datetime")

    df.to_csv(filename, index=False)
    print(f"\n✅ Saved {len(df)} total rows to {filename}")

oct_1_2021_epoch_ms = 1633046400000
oct_1_2025 = 1759276800000

asyncio.run(fetch_volatility_data(oct_1_2021_epoch_ms, oct_1_2025))
