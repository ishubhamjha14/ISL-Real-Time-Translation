"""
tts_service.py
Phase 5: Text & Speech Synthesis (TTS) Integration for Indian Sign Language (ISL).

Refactored with Dedicated Background Worker Thread and Windows COM Multi-threading:
1. Fixes Windows SAPI5 COM engine freeze / loop lockup across subsequent speech calls:
   - Initializes pythoncom.CoInitialize() in the dedicated background thread.
   - Fresh pyttsx3 engine initialization, playback, and teardown per request.
2. Asynchronous Non-Blocking Execution:
   - Dedicated daemon background worker thread consuming an internal speech queue.
   - Never blocks OpenCV video capture, model inference, or UI rendering.
   - Queue size limit (< 2) prevents backlog buildup during rapid signing.
3. Cooldown & Debounce Engine:
   - Prevents repetitive spamming of the same gloss.
4. Fluent Sentence Generation:
   - Maps isolated sign glosses to fluent conversational English sentences.
   - Graceful fallback to natural pronunciation of raw glosses.
"""

import os
import sys
import time
import queue
import threading
from pathlib import Path
from typing import Dict, Optional

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import pyttsx3
except ImportError:
    conda_py = r"C:\Users\abhis\anaconda3\anaconda\envs\isl_py310\python.exe"
    if os.path.exists(conda_py) and sys.executable.lower() != conda_py.lower():
        import subprocess
        result = subprocess.run([conda_py] + sys.argv, cwd=str(project_root))
        sys.exit(result.returncode)
    raise

try:
    import pythoncom
except ImportError:
    pythoncom = None


# Comprehensive 37-Gloss Fluent Translation Dictionary
DEFAULT_GLOSS_TO_TEXT: Dict[str, str] = {
    "nice": "Have a nice day!",
    "atm": "Where is the nearest ATM?",
    "bag": "Please carry this bag.",
    "bank": "I need to visit the bank.",
    "bathroom": "Where is the bathroom?",
    "beautiful": "That is very beautiful!",
    "bedroom": "This is my bedroom.",
    "blind": "The person is visually impaired.",
    "book": "Please open the book.",
    "curved": "The path ahead is curved.",
    "deaf": "I communicate in sign language.",
    "female": "She is a female.",
    "flat": "The surface is completely flat.",
    "fly": "Birds can fly high.",
    "goodmorning": "Good morning, have a great day!",
    "happy": "I am feeling very happy!",
    "hello": "Hello, how are you?",
    "hii": "Hi there!",
    "home": "Let us go home.",
    "hospital": "Please take me to the hospital.",
    "key": "Here is the key.",
    "kitchen": "The food is being prepared in the kitchen.",
    "lock": "Please lock the door.",
    "loud": "The sound is too loud.",
    "male": "He is a male.",
    "mall": "Let us visit the shopping mall.",
    "namaste": "Namaste! Welcome.",
    "paint": "I am going to paint.",
    "photograph": "Please take a photograph.",
    "poor": "They are in need of assistance.",
    "quiet": "Please maintain silence.",
    "rich": "They are very wealthy and rich.",
    "sad": "I am feeling sad today.",
    "telephone": "Please answer the telephone.",
    "thick": "The document is quite thick.",
    "thin": "The paper is very thin.",
    "ugly": "That appears unpleasant.",
}


class TTSService:
    """
    Dedicated Non-blocking Speech Synthesizer Service managing asynchronous speech output
    with Windows COM apartment initialization, fresh SAPI5 engine per request,
    sentence translation, debouncing, and thread safety.
    """

    def __init__(
        self,
        cooldown_seconds: float = 1.5,
        rate: int = 150,
        volume: float = 1.0,
        gloss_map: Optional[Dict[str, str]] = None,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.rate = rate
        self.volume = volume
        self.gloss_map = gloss_map if gloss_map is not None else DEFAULT_GLOSS_TO_TEXT

        # Threading and Queues
        self.speech_queue = queue.Queue()
        self._last_spoken_gloss = ""
        self._last_spoken_time = 0.0

        # State tracking for UI overlay / HUD
        self.is_speaking = False
        self.current_spoken_text = ""
        self.last_spoken_display_time = 0.0

        # Start dedicated background worker daemon thread
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()

    def _process_queue(self):
        """Dedicated background worker thread handling Windows SAPI5 COM engine lifecycle."""
        # Initialize COM library for this background thread on Windows
        if pythoncom:
            try:
                pythoncom.CoInitialize()
            except Exception:
                pass

        while True:
            text = self.speech_queue.get()
            if text is None:
                self.speech_queue.task_done()
                break

            try:
                self.is_speaking = True
                self.current_spoken_text = text
                self.last_spoken_display_time = time.time()

                # Fresh engine initialization per speech request avoids SAPI5 loop lockups
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as e:
                print(f"[TTS Error] {e}")
            finally:
                self.is_speaking = False
                self.speech_queue.task_done()

        if pythoncom:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

    def translate_gloss_to_text(self, gloss: str) -> str:
        """Translates an isolated gloss into a natural fluent sentence if available."""
        clean_key = gloss.strip().lower()
        if clean_key in self.gloss_map:
            return self.gloss_map[clean_key]
        return gloss.strip().capitalize()

    def speak(self, text: str, force: bool = False) -> bool:
        """
        Non-blocking call to queue speech.
        Translates gloss to fluent sentence if recognized and debounces repetition.

        Args:
            text: Gloss name (e.g., 'namaste', 'atm') or raw sentence string.
            force: If True, bypasses cooldown checks.

        Returns:
            True if queued, False if blocked by debounce or queue capacity.
        """
        if not text or text == "---":
            return False

        clean_text = text.strip()
        clean_lower = clean_text.lower()
        now = time.time()

        # Debounce: check if same gloss is requested before cooldown expires
        if not force:
            if clean_lower == self._last_spoken_gloss and (now - self._last_spoken_time) < self.cooldown_seconds:
                return False

        # Translate gloss to fluent sentence if recognized
        phrase = self.translate_gloss_to_text(clean_lower)

        self._last_spoken_gloss = clean_lower
        self._last_spoken_time = now

        # Avoid piling up backlog if user makes multiple gestures rapidly
        if self.speech_queue.qsize() < 2:
            self.speech_queue.put(phrase)
            return True
        return False

    def get_active_spoken_phrase(self, display_duration: float = 3.0) -> Optional[str]:
        """
        Returns the text currently being spoken or recently spoken for UI overlay display.
        """
        if self.is_speaking:
            return self.current_spoken_text
        if (time.time() - self.last_spoken_display_time) < display_duration and self.current_spoken_text:
            return self.current_spoken_text
        return None

    def stop(self):
        """Stops the background worker thread gracefully."""
        try:
            self.speech_queue.put(None)
            if self.worker_thread.is_alive():
                self.worker_thread.join(timeout=1.5)
        except Exception:
            pass


# Alias for backward compatibility with existing imports across ISL pipeline
SpeechSynthesizer = TTSService


if __name__ == "__main__":
    print("=== Testing ISL TTSService (Sequential SAPI5 COM Audio) ===")

    tts = TTSService(cooldown_seconds=1.0)

    test_glosses = ["namaste", "atm", "Nice", "bank"]

    print("\nSimulating sequential recognition stream:")
    for gloss in test_glosses:
        queued = tts.speak(gloss)
        translated = tts.translate_gloss_to_text(gloss)
        status = f"QUEUED -> '{translated}'" if queued else "SKIPPED/BLOCKED"
        print(f"Gloss: {gloss:10s} | {status}")
        # Wait for audio to speak to verify consecutive audio works
        time.sleep(2.5)

    print("\nWaiting for audio queue to finish playing...")
    time.sleep(1.0)

    tts.stop()
    print("=== TTS Service Test Completed Successfully ===")
