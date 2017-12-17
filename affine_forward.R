affine_forward <- function(A_prev, W, b){
  
  # Implement the linear part of a layer's forward propagation.
  # 
  # Arguments:
  # A -- activations from previous layer (or input data): (number of examples, size of previous layer)
  # W -- weights matrix: numpy array of shape (size of previous layer, size of current layer)
  # b -- bias vector, numpy array of shape (1, size of the current layer)
  # 
  # Returns:
  # Z -- the input of the activation function, also called pre-activation parameter 
  # cache -- a R list containing "A", "W" and "b" ; stored for computing the backward pass efficiently
  
  Z <- sweep(A_prev %*% W, 2, b, '+')
  cache <- list(A_prev = A_prev, W = W, b = b)
  list(Z = Z, cache = cache)
}


# WW = matrix(3,4,3)
# AA = matrix(2,10,4)
# bb = matrix(1,1,3)
# affine_output <- affine_forward(AA,WW,bb)