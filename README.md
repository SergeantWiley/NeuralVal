
# Neural Val

A fast and easy small library for exploring or developing proof of concept machine learning models. With flexibility and high modularity, it allows anyone to save hundreds of line of standarzied and common code and pass straight to data quality and annotating

## Install

Clone the repositorty

```
git clone https://github.com/SergeantWiley/NeuralVal.git
```

Enter the director either manualy or through terminal

```
cd NeuralVal
```

The folder can be renamed to the real product or application name if desirable

For libraries

```
pip install -r requirements.txt
```

Note that torch or torch vision is not included to allow flexibility for GPU usage. For NIVIDIA GPUs, install Cuda. For AMD, its ROCm. They can be found [here](https://pytorch.org/). Note that ROCm only works for linux so a linux OS needs to be installed in order to run ROCm

## Usage
The example.py script gives a demonstration and is the base for an entire ML Pipeline however it assumes the data was already annotated

For annotating, CVAT.AI was used and the code was customized for CVAT annotation data. If CAVT.AI was used for your case, then the XMLAnnoationExtract.py file can be used and it will export the data into a csv file in the required format. Speaking required format, this is the required format for csv

`image_name,xmin,ymin,xmax,ymax,Label`

This format is required for the Neural Val custom dataset class to work. To load such a dataset, use the code

```python
dataset,train_loader = modelDataset.load_dataset('MaskImages','annotations.csv')
```

If the model is not being fine tuned after prior training, then use

```python
model = modelArch.PreTrainedArch(GPU=True)
```

GPU by defualt is enabled but if its not avaliable then it will automatically switch to CPU. To force CPU usage, set GPU to False. 

If a model is already fine tuned from a PreTrainedArch such as FasterRCNN then it can be loaded for additional fine tuning to reduce training time by using

```python
model = modelArch.FineTundedArch('fasterrcnn_model.pth',GPU=True)
```
Before training, ensure optimizer and epochs are set
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
```
To start training, pass the model and varaibles from above
```python
training.trainLoop(model,train_loader,optimizer,num_epochs,model_save_path='fasterrcnn_model.pth',GPU=True,monitor=True,train=True)
```
Like the model, by defualt GPU is set to True and CPU can be forced by using GPU = False. Since training can take a long time, monitor is set to True to give feedback on whats happening for each iteration. By defualt its set to true but forced silence is optional by setting it to False. In the case that the train loop wants to be disabled within the script, passing ```train=False``` into the training loop will disable it from training.

Once a model is trained, it is saved and can be loaded seperatly however if its in a seperate file, the PreTrainedArch has to be loaded before loading the new weights
```python
model = modelArch.PreTrainedArch(GPU=True) 
model = utilities.postTrain.load_model('fasterrcnn_model.pth')
```
With this new model, predictions can be made by passing the image and the model. 
```python
predictions = utilities.postTrain.predict('Image628.jpg', model,lower_red = [100, 0, 0], upper_red = [255, 100, 100],mask=True)
```
The example image is below

![image628](https://github.com/SergeantWiley/NeuralVal/assets/86330761/14214270-73e8-4a57-a407-8c339e1b2549)

the lower and upper red are the min and max ranges for what colors will be masked. 

![image](https://github.com/SergeantWiley/NeuralVal/assets/86330761/30c12600-e622-431d-b4e5-481446af2e5f)

Before training, images should be masked but if its desirable that they dont, then set mask to False. An example of what the masked image looks like for both training and post training

![image628](https://github.com/SergeantWiley/NeuralVal/assets/86330761/a9709786-6bb3-4206-87dd-1c26289bbe5d)

To display these predictions, 
```python
print(predictions['boxes']) # Print all the boxes
print(predictions['scores']) # Print all the scores
print(predictions['labels']) # Preint all the labels
utilities.postTrain.display_prediction('Image628.jpg',predictions,threshold=0.7)
```

![Figure_1](https://github.com/SergeantWiley/NeuralVal/assets/86330761/12ac56cc-57f0-4270-b1b0-1f96e0452e0b)

As shown above, a bounding box was created predicted. To set what score should be shown and ones ignored, use the threshold. Above also prints out each part of the output tensor. This can be useful for other things such as score tracking for each image. Additionally, labels and boxes can be used for comparing seperate images to see what data needs to be focused more on. 

## What it shouldn't be used for

Neural Val wasnt designed to be a universal library rather a fast way to get a product up and moving. A lot of code is often standarized and that takes up a lot of code slowing down intital development so Neural Val solves that but Neural Val is not as scalable. 

Neural Val Dataset is restricted by its data format which as of now, contains only one label and its called enemy. Neural Val was designed so the main focus be on the data quality while the code was already made. This reduces intital development a lot but not for final development as data formats and needs may be different format then what Neural Val Dataset class is written in. 
