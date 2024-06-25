from pynput.mouse import Button, Controller
import time
import random

mouse = Controller()

def move_mouse_and_click():
    # Generate a random position
    x = random.randint(0, 1920)  # Assuming a 1920x1080 screen, adjust if needed
    y = random.randint(0, 1080)
    
    # Move the mouse to the random position
    mouse.position = (x, y)
    
    # Perform a click
    mouse.click(Button.left)

    print(f"Moved mouse to ({x}, {y}) and clicked at {time.ctime()}")

try:
    while True:
        move_mouse_and_click()
        time.sleep(5)  # Wait for 2 minutes (120 seconds)
except KeyboardInterrupt:
    print("Script terminated by user.")