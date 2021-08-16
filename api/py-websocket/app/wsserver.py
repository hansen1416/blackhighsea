#!/usr/bin/env python

import logging
import os
import sys
import random
import datetime

import asyncio
import websockets

log_format = "%(levelname)s %(asctime)s - %(message)s"

logging.basicConfig(stream = sys.stdout, format = log_format, level = logging.INFO)
logger = logging.getLogger()

# async def hello(websocket, path):
#     name = await websocket.recv()
#     logger.info(f"< {name}")

#     greeting = f"Hello {name}!"

#     await websocket.send(greeting)
#     logger.info(f"> {greeting}")

async def time(websocket, path):
    while True:
        now = datetime.datetime.utcnow().isoformat() + "Z"
        await websocket.send(now)
        await asyncio.sleep(random.random() * 3)

start_server = websockets.serve(time, "", 4601)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()