pool_backward <- function(dA, cache, mode = "max"){
  
  # Implements the backward pass of the pooling layer
  # 
  # Arguments:
  # dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
  # cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
  # mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
  # 
  # Returns:
  # dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
  
  # Retrieve information from cache
  A_prev <- cache$A_prev
  hparameters <- cache$hparameters
  
  # Retrieve hyperparameters from "hparameters" 
  stride <- hparameters$stride
  f <- hparameters$f
  
  # Retrieve dimensions from A_prev's shape and dA's shape
  m <- dim(A_prev)[1]
  n_H_prev <- dim(A_prev)[2]
  n_W_prev <- dim(A_prev)[3]
  n_C_prev <- dim(A_prev)[4]
  
  m <- dim(dA)[1]
  n_H <- dim(dA)[2]
  n_W <- dim(dA)[3]
  n_C <- dim(dA)[4]

  # Initialize dA_prev with zeros
  dA_prev <- array(0, c(m, n_H_prev, n_W_prev, n_C_prev))
  
  for (i in 1:m) {
    # select training example from A_prev
    a_prev = A_prev[i,,,]
    
    for (h in 1:n_H) {
      for (w in 1:n_W) {
        for (c in 1:n_C) {
          
          # Find the corners of the current "slice"
          vert_start <- (h - 1) * stride + 1
          vert_end <- vert_start + f - 1
          horiz_start <- (w - 1) * stride + 1
          horiz_end <- horiz_start + f - 1
          
          # Compute the backward propagation in both modes.
          if(mode == "max"){
            
            # Use the corners and "c" to define the current slice from a_prev
            a_prev_slice <- a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
            # Create the mask from a_prev_slice
            mask <- create_mask_from_window(a_prev_slice)
            # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA)
            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] <- dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] + mask * dA[i,h,w,c]
            
          } else if(mode == "average"){
            # Get the value a from dA
            da = dA[i,h,w,c]
            # Define the shape of the filter as fxf 
            shape = c(f,f)
            # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. 
            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] <- dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] + distribute_value(da, shape)
            
          }
        }
      }
    }
  }
  
  return(dA_prev)
}


# set.seed(1)
# A_prev <- array(rnorm(5*5*3*2),c(5, 5, 3, 2))
# hparameters <- list(stride=1, f=2)
# A <- pool_forward(A_prev, hparameters)$A
# cache <- pool_forward(A_prev, hparameters)$cache
# dA <- array(rnorm(5*4*2*2), c(5, 4, 2, 2))
# dA_prev <- pool_backward(dA, cache, mode = "max")
# dA_prev <- pool_backward(dA, cache, mode = "average")