import asyncio
import aiohttp
import sys
import json
import os
import time
from pathlib import Path

# URL configuration
BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/progress"

async def verify_progress():
    async with aiohttp.ClientSession() as session:
        # 1. Upload file
        print("Uploading sample.mov...")
        file_path = Path("sample.mov")
        if not file_path.exists():
            print("Error: sample.mov not found.")
            return

        data = aiohttp.FormData()
        data.add_field('file',
                       open(file_path, 'rb'),
                       filename='sample.mov',
                       content_type='video/mp4')

        async with session.post(f"{BASE_URL}/api/transcribe", data=data) as resp:
            if resp.status != 200:
                print(f"Error uploading file: {await resp.text()}")
                return
            result = await resp.json()
            file_id = result["file_id"]
            print(f"Upload successful. File ID: {file_id}")

        # 2. Connect to WebSocket
        print(f"Connecting to WebSocket for {file_id}...")
        async with session.ws_connect(f"{WS_URL}/{file_id}") as ws:
            print("Connected to WebSocket.")

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    print(f"Progress: {data['step']} - {data['progress']}% - {data['message']}")

                    if data['step'] == 'completed':
                        print("Transcription completed successfully!")
                        break
                    if data['step'] == 'error':
                        print(f"Transcription failed: {data['error']}")
                        break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print('ws connection closed with exception %s', ws.exception())
                    break

if __name__ == "__main__":
    # Ensure backend is running before running this script
    try:
        asyncio.run(verify_progress())
    except KeyboardInterrupt:
        print("Verification stopped.")
