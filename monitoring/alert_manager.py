from pathlib import Path

import pygame


class AlertManager:

    def __init__(self, sound_path):

        self.sound = None
        self.channel = None
        self.active = False

        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()

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

        except (pygame.error, OSError) as error:

            print("Audio initialization error:", error)

    def update(self, should_sound):

        if (
            should_sound
            and not self.active
            and self.sound is not None
        ):

            try:
                self.channel = self.sound.play(loops=-1)

                if self.channel is not None:
                    self.active = True
                    print("DROWSINESS ALARM STARTED")

            except pygame.error as error:
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

        except pygame.error:
            pass

        self.channel = None
        self.active = False

    def close(self):

        self.stop()

        if pygame.mixer.get_init():
            pygame.mixer.quit()