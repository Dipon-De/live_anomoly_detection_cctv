from flask import Flask, Response, render_template
import cv2
import asyncio
import websockets
import threading
import json
from detector import ViolenceDetector
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Initialize detector
DETECTOR = ViolenceDetector(
    model_path=os.getenv("MODEL_PATH"),
    telegram_token=os.getenv("TELEGRAM_TOKEN"),
    chat_id=int(os.getenv("CHAT_ID"))
)

clients = set()  # WebSocket clients
camera = cv2.VideoCapture(0)  # Global camera instance

async def notify_clients(message):
    """Send WebSocket notifications to connected clients."""
    if clients:
        await asyncio.gather(*[client.send(json.dumps(message)) for client in clients])

class WebApp:
    def __init__(self, detector):
        self.detector = detector
        self.running = True  # Control loop execution

    def generate_frames(self):
        global camera
        while self.running:
            success, frame = camera.read()
            if not success:
                break

            confidence = self.detector.process_frame(frame)

            text = f"Confidence: {confidence:.2f}%"
            color = (0, 255, 0) if confidence < 50 else (0, 0, 255)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if confidence > 90:
                print(f"🚨 Alert Triggered! Confidence: {confidence:.2f}%")  # Debug

                # Send alert asynchronously
                threading.Thread(target=self.detector.send_alert, args=(frame, confidence), daemon=True).start()

                # Notify WebSocket clients properly
                loop = asyncio.new_event_loop()
                loop.run_until_complete(notify_clients({"alert": True, "confidence": confidence}))
                loop.close()

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def stop(self):
        """Stop video processing."""
        self.running = False
        camera.release()
        print("🔴 Camera stopped.")

webapp = WebApp(DETECTOR)  # Create web app instance

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(webapp.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    webapp.stop()
    return "Camera Stopped"

async def websocket_handler(websocket, path):
    clients.add(websocket)
    try:
        async for message in websocket:
            pass
    finally:
        clients.remove(websocket)

def start_websocket_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = websockets.serve(websocket_handler, '0.0.0.0', 5678)
    loop.run_until_complete(server)
    loop.run_forever()

if __name__ == '__main__':
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)).start()
    threading.Thread(target=start_websocket_server, daemon=True).start()
