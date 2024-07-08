from NeuralVal import *
from NVUtilities import *
dataset,train_loader = modelDataset.load_dataset('MaskImages','annotations.csv') #Load a dataset
model = modelArch.PreTrainedArch(GPU=True) # Load a pretrained Model 
model = modelArch.FineTundedArch('fasterrcnn_model.pth',GPU=True) # Load a .pth model that wants to be refined
optimizer = optim.Adam(model.parameters(), lr=0.001) # Set an optimizer
num_epochs = 10 # Set a number of epochs
training.trainLoop(model,train_loader,optimizer,num_epochs,model_save_path='fasterrcnn_model.pth',GPU=True,monitor=True,train=False) # Run the the Training Loop
model = utilities.postTrain.load_model('fasterrcnn_model2.pth') # Load a model that wants to be used for predictions
predictions = utilities.postTrain.predict('Image628.jpg', model,lower_red = [100, 0, 0], upper_red = [255, 100, 100],mask=True) # Get predictions with an image
print(predictions['boxes']) # Print all the boxes
print(predictions['scores']) # Print all the scores
print(predictions['labels']) # Preint all the labels
utilities.postTrain.display_prediction('Image628.jpg',predictions,threshold=0.7) # Display the results

