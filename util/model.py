import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None

    N = x.shape[0]
    x_reshape = x.reshape(N,-1)
    
    out = x_reshape.dot(w)+ b

    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    N = x.shape[0]
    x_flat = x.reshape(N,-1)
    dx = dout.dot(w.T).reshape(x.shape)

    dw = np.dot(x_flat.T,dout)
    db = np.sum(dout,axis=0)


    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None


    out = np.maximum(0,x)

    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache


    dx = dout * (x>0)


    return dx

def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db



def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None


    N = x.shape[0]
    C = x.shape[1]
    s_y = x[np.arange(N),y]
    loss_mat = np.maximum(0,x+1-s_y.reshape(N,-1))
    loss_mat[np.arange(N),y] = 0
    loss = np.sum(loss_mat)/N 

    dx = np.zeros([N,C])
    dx[loss_mat>0] = 1
    dx[np.arange(N),y] = -dx.sum(axis=1)
    dx /= N



    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None


    N = x.shape[0]
    C = x.shape[1]
    pre_prob_x = np.exp(x)
    prob_x = pre_prob_x/np.reshape(np.sum(pre_prob_x,axis=1),(N,-1))

    loss_1shot = -np.log(prob_x[np.arange(N),y])
    loss = np.sum(loss_1shot)/N


    dx = prob_x.copy()
    dx[np.arange(N),y] -= 1

    dx /= N

    return loss, dx




class SimpleNeuralNetwork:

    def __init__(
        self,
        input_dim=16,
        hidden_dim=20,
        num_classes=2,
        weight_scale=1e-3,
        reg=0.0, ):

        self.params = {}
        self.reg = reg
        # Initialize weights
        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None

        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']


        hidden_layer, cache_hidden = affine_relu_forward(X, W1, b1)


        scores, cache_scores = affine_forward(hidden_layer, W2, b2)

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}


   
        data_loss, dscores = softmax_loss(scores, y)


        reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss = data_loss + reg_loss



        dx2, dW2, db2 = affine_backward(dscores, cache_scores)
        grads['W2'] = dW2 + self.reg * W2  # Add regularization gradient
        grads['b2'] = db2

        # Backward pass for the 1st layer
        dx1, dW1, db1 = affine_relu_backward(dx2, cache_hidden)
        grads['W1'] = dW1 + self.reg * W1  # Add regularization gradient
        grads['b1'] = db1


        return loss, grads





