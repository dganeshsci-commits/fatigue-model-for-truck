"""
Raspberry Pi 5 Hardware GPIO Alert Controller.
Supports Active/Passive Buzzer and Visual Warning LED using gpiozero / gpiod.
Fully compatible with Raspberry Pi 5 RP1 southbridge architecture.
Provides safe mock fallback for development on non-Pi systems.
"""

import threading
import time
import sys


class GPIOAlertController:
    """
    Manages physical cabin audio-visual alerts (buzzer + LED) for the truck driver.
    """

    def __init__(self, buzzer_pin: int = 17, led_pin: int = 27, enabled: bool = True):
        self.buzzer_pin = buzzer_pin
        self.led_pin = led_pin
        self.enabled = enabled
        self.current_alarm = "NONE"
        self._thread = None
        self._stop_event = threading.Event()
        self.is_hardware_available = False

        self.buzzer = None
        self.led = None

        if not self.enabled:
            return

        try:
            # Check for Raspberry Pi OS environment with gpiozero
            from gpiozero import Buzzer, LED
            self.buzzer = Buzzer(self.buzzer_pin)
            self.led = LED(self.led_pin)
            self.is_hardware_available = True
            print(f"[GPIO] Hardware alerts initialized on RPi 5 (Buzzer: GPIO{buzzer_pin}, LED: GPIO{led_pin})")
        except Exception as err:
            self.is_hardware_available = False
            # Fall back to console simulation
            # print(f"[GPIO] Hardware GPIO not available ({err}). Running in software simulation mode.")

    def trigger(self, alarm_level: str):
        """
        Sets the active alarm level: 'NONE', 'LOW', 'MEDIUM', 'HIGH'
        """
        if alarm_level == self.current_alarm:
            return

        self.current_alarm = alarm_level
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.3)

        self._stop_event.clear()

        if alarm_level == "NONE":
            self._turn_off()
            return

        # Start background alarm pattern thread
        self._thread = threading.Thread(target=self._run_pattern, args=(alarm_level,), daemon=True)
        self._thread.start()

    def _turn_off(self):
        if self.is_hardware_available:
            try:
                self.buzzer.off()
                self.led.off()
            except Exception:
                pass

    def _run_pattern(self, level: str):
        while not self._stop_event.is_set():
            if level == "LOW":
                # Single gentle LED pulse (Attention)
                if self.is_hardware_available:
                    self.led.on()
                time.sleep(0.15)
                if self.is_hardware_available:
                    self.led.off()
                time.sleep(1.5)

            elif level == "MEDIUM":
                # Double warning chirp (Drowsy / Yawn)
                for _ in range(2):
                    if self._stop_event.is_set():
                        break
                    if self.is_hardware_available:
                        self.buzzer.on()
                        self.led.on()
                    time.sleep(0.12)
                    if self.is_hardware_available:
                        self.buzzer.off()
                        self.led.off()
                    time.sleep(0.1)
                time.sleep(1.0)

            elif level == "HIGH":
                # Rapid critical alarm pattern (Microsleep / Prolonged eye closure)
                if self.is_hardware_available:
                    self.buzzer.on()
                    self.led.on()
                time.sleep(0.2)
                if self.is_hardware_available:
                    self.buzzer.off()
                    self.led.off()
                time.sleep(0.1)

        self._turn_off()

    def cleanup(self):
        """Releases GPIO pins cleanly on program exit."""
        self._stop_event.set()
        self._turn_off()
        if self.is_hardware_available:
            try:
                self.buzzer.close()
                self.led.close()
            except Exception:
                pass
