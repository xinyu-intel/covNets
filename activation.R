sigmoid <- function(Z){
  
  # Implements the sigmoid activation in numpy
  # 
  # Arguments:
  # Z -- array of any shape
  # 
  # Returns:
  # A -- output of sigmoid(z), same shape as Z
  # cache -- returns Z as well, useful during backpropagation
  
  A <- 1/(1+exp(-Z))
  cache <- Z
  list(A = A, cache = cache)
}

sigmoid_backward <- function(dA, cache){
  
  # Implement the backward propagation for a single SIGMOID unit.
  # 
  # Arguments:
  # dA -- post-activation gradient, of any shape
  # cache -- 'Z' where we store for computing backward propagation efficiently
  # 
  # Returns:
  # dZ -- Gradient of the cost with respect to Z
  
  Z <- cache
  
  s <- 1/(1 + exp(-Z))
  dZ <- dA * s * (1-s)
  
  return(dZ)
}

relu <- function(Z){
  
  # Implement the RELU function.
  # 
  # Arguments:
  # Z -- Output of the linear layer, of any shape
  # 
  # Returns:
  # A -- Post-activation parameter, of the same shape as Z
  # cache -- a R list containing "A" ; stored for computing the backward pass efficiently
  
  A <- pmax(Z, 0)
  cache <- Z
  list(A = A, cache = cache)
}

relu_backward <- function(dA, cache){
  
  # Implement the backward propagation for a single RELU unit.
  # 
  # Arguments:
  # dA -- post-activation gradient, of any shape
  # cache -- 'Z' where we store for computing backward propagation efficiently
  # 
  # Returns:
  # dZ -- Gradient of the cost with respect to Z
  
  Z <- cache
  dZ <- array(dA, dim = dim(dA))
  dZ[Z < 0] <- 0
  
  return(dZ)
}

