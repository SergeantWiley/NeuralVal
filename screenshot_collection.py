import os
import time
from pynput import mouse, keyboard
from PIL import ImageGrab

# Ensure the directory exists
output_dir = 'dataset/RawImages'
os.makedirs(output_dir, exist_ok=True)
press = 0
taking_screenshots = False
def on_click(x, y, button, pressed):
    global press
    # Take action on mouse button release
    if taking_screenshots and button == mouse.Button.left and not pressed:
        # Capture the screen
        screenshot = ImageGrab.grab()
        # Create a unique filename
        #timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(output_dir, f'image{press}.png')
        press += 1
        # Save the screenshot
        screenshot.save(filename)
        print([f'Screenshot saved: {press}'])

# Set up the listener]
def on_press(key):
    global taking_screenshots
    try:
        if key.char == '[':
            taking_screenshots = True
            print('Started taking screenshots')
        elif key.char == ']':
            taking_screenshots = False
            print('Stopped taking screenshots')
    except AttributeError:
        # Handle special keys
        pass

# Set up the listeners
mouse_listener = mouse.Listener(on_click=on_click)
keyboard_listener = keyboard.Listener(on_press=on_press)

print('Listening for key presses and mouse clicks...')
keyboard_listener.start()
mouse_listener.start()
keyboard_listener.join()
mouse_listener.join()

