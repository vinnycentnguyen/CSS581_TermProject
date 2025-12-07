import asyncio
import websockets
import json
import pandas as pd
import time



class BtcIVFetcher:
    def __init__(self, start, end, output_file):
        self.start_time = start
        self.end_time = end
        self.output_file = output_file
        self.api = "wss://www.deribit.com/ws/api/v2"
    

    async def get_data(self):
        all_data = []

        resolution_ms = 3600 * 1000
        max_points = 1000

        async with websockets.connect(self.api) as ws:
            current_start = self.start_time
            while current_start < self.end_time:
                current_end = min(current_start + (max_points * resolution_ms), self.end_time)
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
                candles = resp["result"]["data"]
                all_data.extend(candles)
                last_ts = candles[-1][0]
                current_start = last_ts + resolution_ms
                await asyncio.sleep(0.1)
        df = pd.DataFrame(all_data, columns=["timestamp_ms", "open", "high", "low", "close"])
        df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
        df = df[["datetime", "open", "high", "low", "close"]].sort_values("datetime")

        df.to_csv(self.output_file, index=False)

    def run(self):
        asyncio.run(self.get_data())
