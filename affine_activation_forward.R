affine_activation_forward <- function(A_prev, W, b, activation = "relu"){
  
  # Implement the forward propagation for the LINEAR->ACTIVATION layer
  # 
  # Arguments:
  # A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
  # W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
  # b -- bias vector, numpy array of shape (size of the current layer, 1)
  # activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
  # 
  # Returns:
  # A -- the output of the activation function, also called the post-activation value 
  # cache -- a R list containing "linear_cache" and "activation_cache";
  # stored for computing the backward pass efficiently
  
  if (activation == "sigmoid") {
    affine_forward_output <- affine_forward(A_prev, W, b)
    Z <- affine_forward_output$Z
    affine_cache <- affine_forward_output$cache
    sigmoid_output <- sigmoid(Z)
    A <- sigmoid_output$A
    activation_cache <- sigmoid_output$cache
  } else if (activation == "relu") {
    affine_forward_output <- affine_forward(A_prev, W, b)
    Z <- affine_forward_output$Z
    affine_cache <- affine_forward_output$cache
    relu_output <- relu(Z)
    A <- relu_output$A
    activation_cache <- relu_output$cache
  } else if(activation == "NULL"){
    affine_forward_output <- affine_forward(A_prev, W, b)
    A <- affine_forward_output$Z
    activation_cache <- affine_cache <- affine_forward_output$cache
  }
  
  cache = list(affine_cache = affine_cache, activation_cache = activation_cache)
  
  list(A = A, cache = cache)
}