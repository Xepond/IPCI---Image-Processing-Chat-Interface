import os
import cv2
import numpy as np

from audio_manager import AudioManager
from agent import CommandAgent
from operations import (
    operation_channels,
    operation_detect_edges,
    operation_flip,
    operation_gaussian_blur,
    operation_grayscale,
    operation_inverse,
    operation_laplacian_sharpener,
    operation_shift,
)

IMAGE_PATH = os.getenv("IMAGE_PATH", "lenna.jpeg")
MAX_SHIFT_PIXELS = 2000
MAX_PERCENTAGE = 300
MAX_INTENSITY = 99
WINDOW_NAME = "ImageProcessingVoiceAgent"


class ImageProcessor:
    def __init__(self, image_path: str):
        original = cv2.imread(image_path)
        if original is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        self.original = original
        self.current = original.copy()
        self.history = []  # Stack of previous states for undo

    def save_state(self):
        """Save current image state to history (before applying operation)."""
        self.history.append(self.current.copy())

    def undo(self) -> bool:
        """Undo last operation. Returns True if successful, False if nothing to undo."""
        if not self.history:
            return False
        self.current = self.history.pop()
        return True

    def reset(self):
        """Reset image to original state and clear history."""
        self.current = self.original.copy()
        self.history = []

    def show(self):
        cv2.imshow(WINDOW_NAME, self.current)
        cv2.waitKey(1)

    def shift(self, direction: str, pixels: int):
        self.save_state()
        direction_map = {"up": "n", "down": "s", "left": "w", "right": "e"}
        self.current = operation_shift(self.current, pixels, direction_map[direction])

    def change_contrast(self, percentage: int):
        self.save_state()
        alpha = max(0, 1.0 + (percentage / 100.0))
        self.current = cv2.convertScaleAbs(self.current, alpha=alpha, beta=0)

    def change_brightness(self, percentage: int):
        self.save_state()
        # Clamp to -255 to +255 range to prevent silent clipping
        if abs(percentage) > 255:
            percentage = max(-255, min(percentage, 255))
        beta = int((percentage / 100.0) * 255.0)
        self.current = cv2.convertScaleAbs(self.current, alpha=1.0, beta=beta)

    def apply_blur(self, intensity: int):
        self.save_state()
        k = intensity if intensity % 2 == 1 else intensity + 1
        self.current = operation_gaussian_blur(self.current, (k, k))

    def sharpen(self, intensity: int):
        self.save_state()
        # Map 1-99 to safer amplitude range (1.0-2.5) to avoid clipping artifacts
        # This prevents harsh over-sharpening that creates visual artifacts
        amp = 1.0 + (intensity / 99.0) * 1.5
        self.current = operation_laplacian_sharpener(self.current, amplitude=amp)

    def convert_grayscale(self):
        self.save_state()
        self.current = operation_grayscale(self.current)

    def detect_edges(self):
        self.save_state()
        self.current = operation_detect_edges(self.current)

    def split_channel(self, color: str):
        self.save_state()
        color_map = {"red": "r", "green": "g", "blue": "b"}
        self.current = operation_channels(self.current, color_map[color], colored=True)

    def flip(self, axis: str):
        self.save_state()
        axis_map = {"horizontal": "h", "vertical": "v"}
        self.current = operation_flip(self.current, axis_map[axis])

    def invert(self):
        self.save_state()
        self.current = operation_inverse(self.current)


def _valid_range(value, min_value, max_value):
    return value is not None and min_value <= value <= max_value


def safe_int(value):
    """Gelen değerin güvenli bir şekilde tam sayı olup olmadığını kontrol eder."""
    try:
        if value is None:
            return None
        return int(value)
    except (ValueError, TypeError):
        return None

def main():
    try:
        processor = ImageProcessor(IMAGE_PATH)
    except FileNotFoundError as exc:
        print(f"[SYSTEM] {exc}")
        return

    audio = AudioManager()
    agent = CommandAgent()
    
    audio.speak("System initialized. I am ready when you are.")
    processor.show()

    while True:
        print("\n" + "="*50)
        user_input = input("🟢 Dinlemeyi başlatmak için 'ENTER' tuşuna basın (Çıkmak için 'q' yaz): ")
        
        if user_input.lower() == 'q':
            audio.speak("Shutting down the system. Goodbye.")
            break

        user_text = audio.listen_and_transcribe()
        
        if not user_text:
            print("[SİSTEM] Hiçbir ses algılanmadı. Bekleme moduna dönülüyor.")
            continue

        print(f"\n👤 You: {user_text}")
        
        if "exit" in user_text.lower() or "stop" in user_text.lower():
            audio.speak("Shutting down the system. Goodbye.")
            break

        response = agent.process_command(user_text)
        action = response.get("action")

        if action == "shift_image":
            pixels = safe_int(response.get("pixels"))
            direction = response.get("direction")
            if direction not in {"up", "down", "left", "right"}:
                audio.speak("Which direction? Try up, down, left, or right.")
            elif not _valid_range(pixels, 1, MAX_SHIFT_PIXELS):
                audio.speak(f"How many pixels? Say a number from 1 to {MAX_SHIFT_PIXELS}.")
            else:
                processor.shift(direction, pixels)
                processor.show()
                audio.speak(f"Understood. Shifting the image to the {direction} by {pixels} pixels.")
                agent.clear_history()

        elif action == "flip_image":
            axis = response.get("axis")
            if axis not in {"horizontal", "vertical"}:
                audio.speak("Do you want a horizontal or vertical flip?")
            else:
                processor.flip(axis)
                processor.show()
                audio.speak(f"Understood. Applying {axis} flip.")
                agent.clear_history()

        elif action == "invert_image":
            processor.invert()
            processor.show()
            audio.speak("Understood. Inverting the image colors.")
            agent.clear_history()
                
        elif action == "change_contrast":
            percentage = safe_int(response.get("percentage"))
            if not _valid_range(percentage, -MAX_PERCENTAGE, MAX_PERCENTAGE):
                audio.speak("How much should I change the contrast? Say a number between -300 and 300.")
            else:
                processor.change_contrast(percentage)
                processor.show()
                audio.speak(f"Got it. Adjusting the contrast by {percentage} percent.")
                agent.clear_history()

        elif action == "change_brightness":
            percentage = safe_int(response.get("percentage"))
            if not _valid_range(percentage, -MAX_PERCENTAGE, MAX_PERCENTAGE):
                audio.speak("How much should I change the brightness? Say a number between -300 and 300.")
            else:
                processor.change_brightness(percentage)
                processor.show()
                audio.speak(f"Okay, changing the brightness level by {percentage} percent.")
                agent.clear_history()

        elif action == "apply_blur":
            intensity = safe_int(response.get("intensity"))
            if not _valid_range(intensity, 1, MAX_INTENSITY):
                audio.speak("How much blur do you want? Say a number from 1 to 99, or try 'slightly', 'normal', or 'a lot'.")
            else:
                processor.apply_blur(intensity)
                processor.show()
                audio.speak(f"Understood. Applying a blur filter with an intensity of {intensity}.")
                agent.clear_history()

        elif action == "sharpen_image":
            intensity = safe_int(response.get("intensity"))
            if not _valid_range(intensity, 1, MAX_INTENSITY):
                audio.speak("How much sharpening do you want? Say a number from 1 to 99, or try 'slightly', 'normal', or 'a lot'.")
            else:
                processor.sharpen(intensity)
                processor.show()
                audio.speak(f"Got it. Sharpening the image with an intensity of {intensity}.")
                agent.clear_history()

        elif action == "convert_grayscale":
            processor.convert_grayscale()
            processor.show()
            audio.speak("Understood. Converting the image to black and white.")
            agent.clear_history()

        elif action == "detect_edges":
            processor.detect_edges()
            processor.show()
            audio.speak("Okay, applying edge detection filter.")
            agent.clear_history()

        elif action == "undo":
            if processor.undo():
                processor.show()
                audio.speak("Undo: reverted to previous state.")
            else:
                audio.speak("Nothing to undo.")

        elif action == "reset":
            processor.reset()
            processor.show()
            audio.speak("Reset: image restored to original state.")
            agent.clear_history()

        elif action == "split_channel":
            color = response.get("color")
            if color not in ["red", "green", "blue"]:
                audio.speak("Which color channel? Say red, green, or blue.")
            else:
                processor.split_channel(color)
                processor.show()
                audio.speak(f"Understood. Displaying only the {color} color channel.")
                agent.clear_history()

        else:
            # Catch-all for unknown or unclear commands
            if action == "unknown":
                msg = response.get("message", "I didn't understand that. Please be more specific.")
                audio.speak(msg)
            else:
                audio.speak(response.get("message", "I didn't quite catch that command. Can you repeat?"))
            
            # Don't clear history for unknown - user might follow up

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()