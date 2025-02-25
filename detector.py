import cv2
import numpy as np
import pickle
import time
import asyncio
import telegram
from threading import Thread

class ViolenceDetector:
    def __init__(self, model_path, telegram_token, chat_id):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.bot = telegram.Bot(token=telegram_token)
        self.chat_id = chat_id
        self.last_alert_time = 0
        self.alert_cooldown = 5  # Send alert every 60 seconds max

        # Initialize async loop properly
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self.run_loop, daemon=True)
        self.thread.start()

    def run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def send_alert(self, frame, confidence):
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            cooldown_remaining = int(self.alert_cooldown - (current_time - self.last_alert_time))
            print(f"⏳ Cooldown active! Alert skipped. Wait {cooldown_remaining} seconds.")
            return

        alert_image_path = 'alert.jpg'
        cv2.imwrite(alert_image_path, frame)

        message = f"⚠️ VIOLENCE DETECTED! Confidence: {confidence:.2f}%"

        # Use loop.create_task instead of call_soon_threadsafe
        self.loop.call_soon_threadsafe(self.loop.create_task, self._async_send_alert(alert_image_path, message))

        self.last_alert_time = time.time()

    async def _async_send_alert(self, image_path, message):
        try:
            async with self.bot:
                with open(image_path, 'rb') as photo:
                    await self.bot.send_photo(chat_id=self.chat_id, photo=photo, caption=message)
                    print(f"✅ Alert sent! Telegram Response")
        except telegram.error.BadRequest as e:
            print(f"❌ Telegram API Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")

    def process_frame(self, frame):
        frame = cv2.resize(frame, (160, 160))
        frame = frame.astype('float32') / 255.0
        frame = np.expand_dims(frame, axis=0)

        prediction = self.model.predict(frame)
        confidence = float(prediction[0][0]) * 100

        return confidence
