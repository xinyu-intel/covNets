affine_backward <- function(dZ, cache){
  
  # Implement the linear portion of backward propagation for a single layer (layer l)
  # 
  # Arguments:
  # dZ -- Gradient of the cost with respect to the linear output (of current layer l)
  # cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
  # 
  # Returns:
  # dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
  # dW -- Gradient of the cost with respect to W (current layer l), same shape as W
  # db -- Gradient of the cost with respect to b (current layer l), same shape as b
  
  A_prev <- cache$A_prev
  W <- cache$W
  b <- cache$b
  m <- dim(A_prev)[1]
  
  dW <- (t(A_prev) %*% dZ) / m
  db <- (matrix(colSums(dZ),1,dim(b)[2])) / m
  dA_prev <- dZ %*% t(W)
  
  list(dA_prev = dA_prev, dW = dW, db = db)
  
}