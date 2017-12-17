affine_activation_backward <- function(dA, cache, activation){
  
  # Implement the backward propagation for the LINEAR->ACTIVATION layer.
  # 
  # Arguments:
  # dA -- post-activation gradient for current layer l 
  # cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
  # activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
  # 
  # Returns:
  # dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
  # dW -- Gradient of the cost with respect to W (current layer l), same shape as W
  # db -- Gradient of the cost with respect to b (current layer l), same shape as b
  
  affine_cache <- cache$affine_cache
  activation_cache <- cache$activation_cache
  
  if (activation == "relu") {
    dZ <- relu_backward(dA, activation_cache)
  } else if (activation == "sigmoid") {
    dZ <- sigmoid_backward(dA, activation_cache)
  } else if (activation == "NULL") {
    dZ <- dA
  }
  
  affine_backward_output <- affine_backward(dZ, affine_cache)
  dA_prev <- affine_backward_output$dA_prev
  dW <- affine_backward_output$dW
  db <- affine_backward_output$db
  
  list(dA_prev = dA_prev, dW = dW, db =db)
}