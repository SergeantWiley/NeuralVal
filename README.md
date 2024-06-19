
# Neural Val

If you haven't read on ChessNet, its recommended to catch up on it. You can read it here: [ChessNet](https://github.com/SergeantWiley/ChessNet)

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

![image108](https://github.com/SergeantWiley/NeuralVal/assets/86330761/9cc36147-da17-496c-815c-6619cccfc9f7)

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

Through this proccess, anything that falls within this range is consider "red" and anything outside this range is consider not "red". Of course these values can be adjusted depending on what the needs are. But we will discover this range isn't what we want exactly. 

![image108](https://github.com/SergeantWiley/NeuralVal/assets/86330761/8856a14e-410b-4a4f-8671-9d3b2cc7815a)

We have removed quite a lot of things but also added some odd looking shapes but that can be taken to our advantage. Our model doesn't need to learn the colors, rather we want it to learn the shape and the area within and if we zoom in to the enemy, we find some interesting traits

![image](https://github.com/SergeantWiley/NeuralVal/assets/86330761/f87d7967-fffe-41c4-b065-2e397db7a781)

With some basic observation, the highest resolution is the legs and the thighs. In matter a fact, during Data Annotation which will be discussed next, the characters can actually be defined by the legs and thighs *Writer's Note: This was only discovered after annotating the data*

This takes us to our final Data Refining proccess and often the most time consuming and repeative proccess and its Data Annotating

# Data Annotating

This task is so long and extensive, there is an entire job industry for it. Data annotating in object detect is often done by drawing bounding boxes using a data annotating tool. Many tools are free and are great for many objectives. [CVAT.AI](https://www.cvat.ai/) was used in the data annotating proccess.

The most common and simplist annotation for object detection is a bounding box as it has only 6 total columns: `Images, xmin, xmax, ymin, ymax, label`.

![image](https://github.com/SergeantWiley/NeuralVal/assets/86330761/5dc9bc85-368b-45f6-8122-0e9eea2b6f50)

`Example: image11.jpg, 951.23, 200.69, 975.79, 245.93, Enemy`

Most data annotating tools have the same general format just different UI or other unique tools but the concept is exactly the same regardless of the tool. For the sake of consistency, CVAT will be used. When exporting data, it exports as an XML file which needs to be converted into CSV. Using the code below we can extract the 8 mentioned things above but also ignore any images without labels

```python
import xml.etree.ElementTree as ET
import csv

input_xml_file = 'annotations.xml'
output_csv_file = 'annotations.csv'
tree = ET.parse(input_xml_file)
root = tree.getroot()

with open(output_csv_file, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    writer.writerow(['image_name', 'xmin', 'xmax', 'ymin', 'ymax','label'])
    
    for image in root.findall('image'):
        image_name = image.get('name')
        
        for box in image.findall('box'):
            xmin = box.get('xtl')
            xmax = box.get('xbr')
            ymin = box.get('ytl')
            ymax = box.get('ybr')
            label_name = box.get('label')
            writer.writerow([image_name, xmin, ymin, xmax, ymax,label_name])

print(f"Annotations successfully exported to {output_csv_file}")
```

# Data Importing

Unlike ChessNet, a nueral network architecture wont be defined, rather a custom dataset called NeuralVal as we will be using a Pretrained Model called FasterNet Version 50
```python
class NueralVal(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        boxes = self.annotations.iloc[idx, 1:5].values.astype('float').reshape(-1, 4)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        if self.transform:
            image = self.transform(image)
        target = {'boxes': boxes, 'labels': labels}
        return image, target
```
At the start of the class, we declared the image director, annotations, and any optional transformations. 

```python
def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform
```
The get item is used in all custom dataset as it formats the data in the same way torch wants it. Specifcally torchvision. We start by creating the img path then load all images using Pillow. For the boxes, we can index the columns 1 through 5 inside the annotations file. We dont have any labels so we will stick with 0. Finally, we create a target which contains the boxes and the labels. 

```python
img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
image = Image.open(img_path).convert("RGB")
boxes = self.annotations.iloc[idx, 1:5].values.astype('float').reshape(-1, 4)
boxes = torch.as_tensor(boxes, dtype=torch.float32)
labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
target = {'boxes': boxes, 'labels': labels}
```

With this dataset, we can then pass it into the pretrained model. The reason for a pretrained over a untrained is due to a couple of advantages. One mainly is the lower computation power needed over the computation power needed for a empty model. A few more reasons below:

* Starting somewhere instead of nowhere: Our pretrained may not be exactly what we want but we can finetune the model to meet our needs

* Low computation: A pretrained doesnt need memory at the level of an empty model. For instance, using a pretrained needed around 7GB of RAM while a empty model needed 52GB of RAM

* Built in optimization: Pretrained models have been developed and managed over time allowing for models to become more optimized.

While this may seem a promising choice, it does still require extensive computation power. For instance, a RTX 3070 ran for 5 hours to get a confidence of 86 percent but the advantage is in the memory demand. It only demanded 7GB over the 52GB

