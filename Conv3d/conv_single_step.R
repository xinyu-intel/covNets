conv_single_step <- function(a_slice_prev, W, b){
  
  # Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
  # of the previous layer.
  # 
  # Arguments:
  # a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
  # W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
  # b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
  # 
  # Returns:
  # Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
  
  s <- a_slice_prev * W + b
  Z <- sum(s)
  return(Z)
}

set.seed(1)
a_slice_prev <- array(rnorm(4*4*3),c(4,4,3))
W <- array(rnorm(4*4*3),c(4,4,3))
b <- rnorm(1)
Z <- conv_single_step(a_slice_prev, W, b)
Z # -56.07105