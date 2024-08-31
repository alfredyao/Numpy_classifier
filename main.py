import numpy as np
import json
import os
from util import model
from util import trainer

def load_from_json(filename='data.json'):
    """
    Loads the feature matrix X and labels y from a JSON file.

    Parameters:
    - filename: str, the name of the file to load the data from.

    Returns:
    - X: np.array, feature matrix.
    - y: np.array, labels.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
  #  X = np.array(data["features"])
  #  y = np.array(data["labels"])
    
    return data

file_name = os.path.join(os.path.dirname(__file__), 'dataset/traindata')
traindata = load_from_json(filename=file_name)

file_name = os.path.join(os.path.dirname(__file__), 'dataset/evaldata')
evaldata = load_from_json(filename=file_name)

X = np.array(traindata['features'])
#y = np.array(traindata['labels']).astype(int)



nn = model.SimpleNeuralNetwork(input_dim=X.shape[1],hidden_dim=20,num_classes=2,reg=0.1) 
nntrainer = trainer.trainer(model=nn, data=traindata,learning_rate=1e-2,epochs=100,batchsize=50 )

nntrainer.train()

X_eval = np.array(evaldata['features'])
y_eval = np.array(evaldata['labels']).astype(int)

evalloss, evalgrad = nn.loss(X_eval,y_eval)

print(evalloss)

# Initialize and train the network
#nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
#nn.train(X, Y, epochs=1000, learning_rate=0.1)

# Test the network
#predictions = nn.forward(X)
#print("Predictions after training:", predictions)