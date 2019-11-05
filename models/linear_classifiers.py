import numpy as np

class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
              training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
              means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None
            rand_Indices=np.random.choice(num_train,batch_size)
            X_batch=X[rand_Indices]
            y_batch=y[rand_Indices]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            # perform parameter update
            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        return np.argmax(X @ self.W,1)
  
    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """
    def loss(self, X_batch, y_batch, reg):
        """Structured SVM loss function, vectorized implementation.
           Inputs and outputs are the same as svm_loss_naive."""
        """loss = if correct, 0, if not, add the margin if its larger than zero"""
        """The margin for a sample and a class is"""
        n_train = len(X_batch)
        n_feat, n_cls = self.W.shape
        
        scores = X_batch @ self.W
        
        loss = 0
        dW = np.zeros(self.W.shape)
        for i in range(n_train):
            rt_cls_scr = np.argmax(scores[i])
            for j in range(n_cls):
                if j==y_batch[i]: continue
                margin = scores[i][j] - scores[i][y_batch[i]] + 1
                if margin > 0:
                    loss += margin
                    dW[:,j] += X_batch[i,:]
                    dW[:,y_batch[i]] -= X_batch[i,:] 
                    
        loss += reg * np.sum(self.W*self.W)
        dW += 2 * reg * self.W
            
        loss /= n_train
        dW /= n_train
        
        return loss, dW
        
    def loss_correct(self, X_batch, y_batch, reg):
        """Structured SVM loss function, vectorized implementation.
           Inputs and outputs are the same as svm_loss_naive."""
        loss = 0.0
        dW = np.zeros(W.shape) # initialize the gradient as zero
        num_classes = W.shape[1]
        num_datadim = W.shape[0]
        num_train = X.shape[0]
  
        scores = X @ W
        right_class =  scores[range(num_train),y]
        #margin has the same shape as scores, (N,C)
        margin = np.maximum(0,(scores.T - right_class +1).T )  
        margin[range(num_train),y] = 0
        loss = np.sum(margin)/num_train
        loss += reg * np.sum(W * W)
  
        nonzero = (margin>0)*1#*1 turns boolean array to int array
        nonzero[range(num_train),y] = -(np.sum(nonzero, axis=1))
        dW = X.T @ nonzero
        dW /= num_train
        dW += reg*W
  
        return loss, dW


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

