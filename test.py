from NueralVal import *
from NVUtilities import *
dataset,train_loader = modelDataset.load_dataset('MaskImages','annotations.csv')
model = modelArch.FineTundedArch('fasterrcnn_model.pth')
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
training.trainLoop(model,train_loader,optimizer,num_epochs,'fasterrcnn_model.pth',False)
model = utilities.postTrain.load_model('fasterrcnn_model2.pth')
predictions = utilities.postTrain.predict('Image683.jpg', model)
print(predictions['scores'])
utilities.postTrain.display_prediction('Image683.jpg',predictions,0.3)