from pathlib import Path

import os
import sys

# Avoid pygame's startup banner.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# Servers have no sound card; the dashboard plays the alarm in the browser.
if sys.platform.startswith("linux"):
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

try:
    import pygame
except (ImportError, OSError) as error:
    pygame = None
    print(f"Audio module unavailable: {error}")


class AlertManager:
    def __init__(self, sound_path):
        self.sound = None
        self.channel = None
        self.active = False
        self.audio_available = False

        if pygame is None:
            print("Audio alerts disabled. Monitoring can continue without sound.")
            return

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            self.audio_available = True
            sound_path = Path(sound_path)

            if sound_path.is_file():
                self.sound = pygame.mixer.Sound(
                    str(sound_path.resolve())
                )

                print(
                    "Alarm loaded successfully:",
                    sound_path.resolve()
                )
            else:
                print(
                    "WARNING: Alarm file not found:",
                    sound_path.resolve()
                )

        except (pygame.error, OSError, RuntimeError) as error:
            self.audio_available = False
            self.sound = None
            self.channel = None
            print("Audio initialization error:", error)
            print("Continuing without audio alerts.")

    def update(self, should_sound):
        if not self.audio_available or self.sound is None:
            self.active = False
            return

        if should_sound and not self.active:
            try:
                self.channel = self.sound.play(loops=-1)

                if self.channel is not None:
                    self.active = True
                    print("DROWSINESS ALARM STARTED")

            except (pygame.error, OSError, RuntimeError) as error:
                self.active = False
                self.channel = None
                print("Alarm playback error:", error)

        elif not should_sound and self.active:
            self.stop()
            print("Drowsiness alarm stopped.")

    def stop(self):
        try:
            if self.channel is not None:
                self.channel.stop()
            elif self.sound is not None:
                self.sound.stop()

        except (pygame.error, OSError, RuntimeError):
            pass

        self.channel = None
        self.active = False

    def close(self):
        self.stop()

        self.audio_available = False
