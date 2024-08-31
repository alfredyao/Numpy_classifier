import numpy as np

class trainer:

    def __init__(self,model,data,**kwargs):

        self.model = model
        self.X = np.array(data['features'])
        self.y = np.array(data['labels']).astype(int)

        self.learning_rate = kwargs.pop("learning_rate")
        self.epochs = kwargs.pop('epochs')
        self.batchsize = kwargs.pop('batchsize')




    def train(self):
        for epoch in range(self.epochs):
            # Forward pass and compute loss
            
            total = self.X.shape[0]
            iterations = total//self.batchsize
            
            for iteration in range(iterations):
                batch_mask = np.random.choice(total,self.batchsize)


                loss, grads = self.model.loss(self.X[batch_mask],self.y[batch_mask])

                for k,v in self.model.params.items():
                    self.model.params[k] -= self.learning_rate*grads[k]

            

            # Print the loss every 100 epochs
            if epoch % 5 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')