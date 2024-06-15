
# Neural Val

If you haven't read on ChessNet, its recommended to catch up on it. You can read it here: [ChessNet](doc:linking-to-pages#anchor-links)

Neural Val is designed to demonstrate object detection and classification in machine learning. For the dataset, we will use screenshots from the 5v5 FPS game Valorant for our ML Pipeline. Due to the massive amount of computation power needed, it is recommended that if you dont have a RTX 3060, then its best not to train a model. Which is why we will be focusing on the ML Pipeline.

## ML Pipeline

Like the name suggest, the pipeline uses a methodolgy for optimizing ML training. 

This is the general pipeline:

* Data Collection/Mining
* Data Transforming
* Annotating
* Data Importing
* Pretrained models
* Terrifying GPU/CPU consumption

The hardest part of all this is the data refining as it requires an understanding of what we are trying to replicate. While ChessNet goes indepth into the refining proccess but thats nothing compared to Nueral Val. Let the fun begin

# Data Collection

Data Collection starts by effeceintly collecting images or frames that have most the highest likely chance of containing something we want to teach. In this case, we want to have as many frames of a enemy player in Valorant. Downside is that a video would make it easy but many Valorant players can say only few seconds per round actually has an enemy visible more then 3 frames so the best way was to take a screenshot every time an enemy appears by assigning the fire button as the key to take the screenshot.

First is to establish a few variables and importing libraries
```python
import os
from pynput import mouse, keyboard
from PIL import ImageGrab
import threading #For multithreading

output_dir = 'RawImages'
os.makedirs(output_dir, exist_ok=True)
press = 0 #What image number to start on
taking_screenshots = False
```
Then its making the function to take a screenshot
```python
def take_screenshot():
    global press

    screenshot = ImageGrab.grab()
    filename = os.path.join(output_dir, f'image{press}.png')
    press += 1
    screenshot.save(filename)

    print(f'Screenshot saved: {press}')
```
Now its to detect the fire button. There are a few parts that happen very quickly to ensure the highest quality of data collection 

```python
def on_click(x, y, button, pressed):
    global press
    if taking_screenshots and button == mouse.Button.left and not pressed:
        screenshot_thread = threading.Thread(target=take_screenshot)
        screenshot_thread.start()
```
First we check if taking_screenshots is true then we check if button we clicked was the left mouse button (fire button) and if it is not being held down as in valorant, its rare to hold down the fire button for the most common used guns.

While pressing the fire button is nice and all, the button is usally the left mouse button which is used for a lot more then just firing in game so the ability to control when the screenshots to be taken. For the keys, `[` is for starting the program and to stop or pause is `]`

```python
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
        pass
```

Finally, call and intilize everything

``` python
mouse_listener = mouse.Listener(on_click=on_click)
keyboard_listener = keyboard.Listener(on_press=on_press)

print('Listening for key presses and mouse clicks...')
keyboard_listener.start()
mouse_listener.start()
keyboard_listener.join()
mouse_listener.join()

```

The script as the whole uses many small means of ensuring that during the data mining, the least amount of impurities during the collection proccess and those impurities may be screenshots without the objects we want. 

# Data Cleansing

Right now the data is raw format and a lot of impurities and a lot of containiments. Data Cleansing removes these and ensures the integrity is higher before it was cleaned. This may also be refered as Data Refining in some cases. 

This is an example of raw mined data

![image12](https://github.com/SergeantWiley/NeuralVal/assets/86330761/cea683a3-53f3-4e0a-843f-0ab83152afab)

There is quite a lot of useless information for object detection. For example, the mini map isnt needed, the buttom abilities arent needed, the combat report, or any GUI so the best method would to cut the image to a FOV

```python
def crop_center(image, crop_width, crop_height):
    # Get the dimensions of the image
    height, width, _ = image.shape

    center_x, center_y = width // 2, height // 2

    x1 = center_x - crop_width // 2
    x2 = center_x + crop_width // 2
    y1 = center_y - crop_height // 2
    y2 = center_y + crop_height // 2

    # Crop the image
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def process_images(input_dir, output_dir, crop_size):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            # Check if the image was loaded properly
            if image is None:
                print(f"Error: Could not read the image {filename}.")
                continue

            # Crop the center of the image
            cropped_image = crop_center(image, crop_size[0], crop_size[1])

            # Save the cropped image to the output directory
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, cropped_image)
            print(f"Cropped image saved to {output_path}")

# Example usage
input_directory = 'RawImages'
output_directory = 'FOVImages'
crop_size = (1920, 400)  # Width x Height of the crop region
process_images(input_directory, output_directory, crop_size)
```
Playes are most commonly looking at the front for the enemy thus they are likely to be viewed near the middle rows of the image, the bottom can be cut off and the top can be cut off. Using the crop_size, we can choose what the end shape would look like. This is important that this FOV stays conistent

![image12](https://github.com/SergeantWiley/NeuralVal/assets/86330761/b5ba8515-e8f1-48b6-b6ad-c36818077139)

Of course further refining can be done but for now, we will stick with this before moving to the next part. The next part is masking. Like before, there is a lot of code that ensures it goes from RGB to only black and white with a little bit of contrast. 

```python
import cv2
import numpy as np
import os

def red_mask_intensity(image_path, output_dir):
    # Read the original image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read the image {image_path}.")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lower_red = np.array([100, 0, 0])
    upper_red = np.array([255, 100, 100])
    mask = cv2.inRange(image_rgb, lower_red, upper_red)                 
    green_channel = image_rgb[:, :, 1]
    blue_channel = image_rgb[:, :, 2]
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
    gray_mask_float = gray_mask.astype(np.float32)
    gray_mask_float /= 255.0
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, filename + '.jpg')
    cv2.imwrite(output_path, (gray_mask_float * 255).astype(np.uint8))
    print(f"Grayscale intensity mask saved to {output_path}")

input_dir = 'FOVImages'
output_dir = 'MaskImages'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        red_mask_intensity(image_path, output_dir)
```
While it seems like a lot, the most important key things are the

```python
lower_red = np.array([100, 0, 0])
upper_red = np.array([255, 100, 100])
input_dir = 'FOVImages'
output_dir = 'MaskImages'
```
The most important is the lower_red and the upper_red. These dictate what counts as "red" and what doesn't count as "red". Anyone can say the enemies are clearly red but so is the gun and is the combat report in the images above. In matter a fact, everything in the image has some red so exactly classifying what red is cant be done with on value so a range is dictated. That range looks like this:
![image](https://github.com/SergeantWiley/NeuralVal/assets/86330761/30c12600-e622-431d-b4e5-481446af2e5f)

