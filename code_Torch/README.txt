train_CRF takes several parameters from command.

# -optim specifies the gradient descent method, lbfgs by default
# option sag/sgd/lbfgs
th ./train_CRF -optim <sag/lbfgs/sgd>

# -iter sets the maximal iteration number 
th ./train_CRF -iter <n>

# -lambda sets the regularization term
th ./train_CRF -lambda <n>

# -lr specifies the learning rate/step size
th ./train_CRF -lr <n>

# -b sets the number of updates between evaluations on the entire training/testing set
th ./train_CRF -b <n>

# -n specifies the number of words used for training,
# default is 100, if sets -1, use the entire training set
th ./train_CRF -n <n>