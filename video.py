#!/usr/bin/env python3
import asyncio
import json
import threading
import io
import time

import av
import cv2
import websockets 

# -------------------------------
# BufferReader: A file-like object for PyAV
# -------------------------------
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

# -------------------------------
# VideoStream: Manages one video stream
# -------------------------------
class VideoStream:
    def __init__(self, uri, window_name):
        self.uri = uri
        self.window_name = window_name
        self.buffer_reader = BufferReader()
        self.decoder_thread = None

    def start_decoder(self):
        # Start a thread that decodes frames from our buffer.
        def decoder_worker():
            try:
                # Open the container; adjust format if needed.
                container = av.open(self.buffer_reader, format='mp4', mode='r')
                for frame in container.decode(video=0):
                    img = frame.to_ndarray(format='bgr24')
                    cv2.imshow(self.window_name, img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except Exception as e:
                print(f"[{self.window_name}] Decoding error: {e}")
            finally:
                cv2.destroyWindow(self.window_name)
        self.decoder_thread = threading.Thread(target=decoder_worker, daemon=True)
        self.decoder_thread.start()

    async def websocket_worker(self):
        async with websockets.connect(self.uri) as ws:
            print(f"[{self.window_name}] WebSocket connection opened.")
            # Send initial commands
            await ws.send("iq2")
            await ws.send("quality0")

            # Start background ping task
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

            # Process incoming messages
            async for message in ws:
                if isinstance(message, str):
                    print(f"[{self.window_name}] Received text message: {message}")
                elif isinstance(message, bytes):
                    self.buffer_reader.feed(message)

    def close(self):
        self.buffer_reader.close()

# -------------------------------
# List of websocket URIs and window names.
# -------------------------------
STREAMS = [
    {
        "uri": ("wss://ws1.zigdpseatvkmftqo.net/BJ5.1?"
                "JSESSIONID=20AGRCD3c59DjD_gtLPM16if7ZVmv1_KhPgnNSVD4XGrjy1X4SZ5!"
                "98817579-aa178660"),
        "window": "BJ5.1 Stream"
    },
    {
        "uri": ("wss://ws1.zigdpseatvkmftqo.net/BJ22.1-Generic?"
                "JSESSIONID=nhgJ5k6rPfPzAlE6FgjqUMiH1UChat5ak49czYwISMVRx3FBajsz!-1416944353-b2d1fa2d"),
        "window": "BJ22.1 Generic"
    },
    {
        "uri": ("wss://ws1.zigdpseatvkmftqo.net/A33BJ2-Generic?"
                "JSESSIONID=jcsJ6B7QfriEMFc-8hoDG7lzyIb--eFgp24cIANlTHyN1SCkBgZ_!-1475960653-cfc108af"),
        "window": "A33BJ2 Generic"
    },
    {
        "uri": ("wss://ws1.zigdpseatvkmftqo.net/BJ3.1-GENERIC?"
                "JSESSIONID=jcsJ6B7QfriEMFc-8hoDG7lzyIb--eFgp24cIANlTHyN1SCkBgZ_!-1475960653-cfc108af"),
        "window": "BJ3.1 GENERIC"
    }
]

# -------------------------------
# Main function: start decoders and websockets for all streams.
# -------------------------------
async def main():
    video_streams = []
    websocket_tasks = []
    for stream in STREAMS:
        vs = VideoStream(uri=stream["uri"], window_name=stream["window"])
        vs.start_decoder()
        video_streams.append(vs)
        websocket_tasks.append(vs.websocket_worker())
    # Run all websocket workers concurrently.
    await asyncio.gather(*websocket_tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Exiting...")
    # Close all buffer readers.
    for stream in STREAMS:
        # Since we don't have direct references here, if needed add cleanup logic.
        pass
