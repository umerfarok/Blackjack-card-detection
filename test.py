#!/usr/bin/env python3
import asyncio
import json
import threading
import io
import time

import av  # PyAV
import cv2
import websockets
 
# A custom file-like object to serve as a read interface for PyAV.
class BufferReader(io.RawIOBase):
    def __init__(self):
        self.buffer = bytearray()
        self.closed_flag = False
        self.lock = threading.Lock()
        self.data_available = threading.Condition(self.lock)

    def feed(self, data: bytes):
        with self.lock:
            self.buffer.extend(data)
            self.data_available.notify_all()

    def read(self, n=-1):
        with self.lock:
            while not self.buffer and not self.closed_flag:
                self.data_available.wait()
            if not self.buffer:
                return b''
            if n < 0 or n > len(self.buffer):
                n = len(self.buffer)
            result = bytes(self.buffer[:n])
            del self.buffer[:n]
            return result

    def readable(self):
        return True

    def close(self):
        with self.lock:
            self.closed_flag = True
            self.data_available.notify_all()
        super().close()

# Global instance of our buffer reader.
buffer_reader = BufferReader()

# This worker opens the container from our buffer and decodes video frames.
def decoder_worker():
    try:
        # Adjust the 'format' parameter if your stream is not fragmented MP4.
        container = av.open(buffer_reader, format='mp4', mode='r')
        for frame in container.decode(video=0):
            # Convert the frame to a numpy array (BGR format for OpenCV)
            img = frame.to_ndarray(format='bgr24')
            cv2.imshow('Live Video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print("Decoding error:", e)
    finally:
        cv2.destroyAllWindows()

# Start the decoder worker in a separate thread.
def start_decoder():
    threading.Thread(target=decoder_worker, daemon=True).start()

# The websocket worker connects to the stream and feeds binary data into our buffer.
async def websocket_worker():
    # Replace with your actual websocket URI.
    uri = (
        "wss://ws1.zigdpseatvkmftqo.net/BJ5.1?"
        "JSESSIONID=20AGRCD3c59DjD_gtLPM16if7ZVmv1_KhPgnNSVD4XGrjy1X4SZ5!"
        "98817579-aa178660"
    )
    async with websockets.connect(uri) as ws:
        print("WebSocket connection opened.")
        # Send the initial commands.
        await ws.send("iq0")

        # Start a background ping task: send a ping message every 30 seconds.
        async def ping_task():
            counter = 1
            while True:
                await asyncio.sleep(30)
                ping_msg = {
                    "cmd": "ping",
                    "counter": counter,
                    "clientTime": int(time.time() * 1000)
                }
                await ws.send(json.dumps(ping_msg))
                counter += 1

        asyncio.create_task(ping_task())

        # Process incoming messages.
        async for message in ws:
            if isinstance(message, str):
                print("Received text message:", message)
            elif isinstance(message, bytes):
                # Feed binary messages into our buffer so the decoder can read them.
                buffer_reader.feed(message)

if __name__ == "__main__":
    start_decoder()  # Start the video decoder in a background thread.
    try:
        asyncio.run(websocket_worker())
    except KeyboardInterrupt:
        print("Exiting...")
        buffer_reader.close()