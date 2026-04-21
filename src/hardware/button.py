import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class Button:
    """GPIO button with keyboard fallback for development."""

    def __init__(self, gpio_pin: int = 17, use_keyboard_fallback: bool = True):
        self.gpio_pin = gpio_pin
        self.use_keyboard_fallback = use_keyboard_fallback
        self._callback = None
        self._running = False
        self._thread = None
        self._gpio_available = False

        # Try to initialize GPIO
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.gpio_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            self._gpio_available = True
            logger.info(f"GPIO button initialized on pin {self.gpio_pin}")
        except (ImportError, RuntimeError):
            logger.info("GPIO not available, using keyboard fallback (press Enter)")

    def on_press(self, callback: Callable[[], None]) -> None:
        """Register a callback for button press. Starts listening."""
        self._callback = callback
        self._running = True

        if self._gpio_available:
            import RPi.GPIO as GPIO
            GPIO.add_event_detect(
                self.gpio_pin,
                GPIO.FALLING,
                callback=lambda _: self._handle_press(),
                bouncetime=300,
            )
        elif self.use_keyboard_fallback:
            self._thread = threading.Thread(target=self._keyboard_loop, daemon=True)
            self._thread.start()

    def _handle_press(self):
        if self._callback and self._running:
            self._callback()

    def _keyboard_loop(self):
        while self._running:
            try:
                input()  # Wait for Enter
                self._handle_press()
            except EOFError:
                break

    def cleanup(self) -> None:
        self._running = False
        if self._gpio_available:
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup(self.gpio_pin)
            except (ImportError, RuntimeError):
                pass
