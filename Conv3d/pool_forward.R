pool_forward <- function(A_prev, hparameters, mode = "max"){
  
  # Implements the forward pass of the pooling layer
  # 
  # Arguments:
  # A_prev -- Input data, array of shape (m, n_H_prev, n_W_prev, n_C_prev)
  # hparameters -- R list containing "f" and "stride"
  # mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
  # 
  # Returns:
  # A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
  # cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
  
  # Retrieve dimensions from A_prev's shape
  m <- dim(A_prev)[1]
  n_H_prev <- dim(A_prev)[2]
  n_W_prev <- dim(A_prev)[3]
  n_C_prev <- dim(A_prev)[4]
  
  # Retrieve information from "hparameters"
  stride <- hparameters$stride
  f <- hparameters$f
  
  # Define the dimensions of the output
  n_H = floor(1 + (n_H_prev - f) / stride)
  n_W = floor(1 + (n_W_prev - f) / stride)
  n_C = n_C_prev
  
  A = array(0, c(m, n_H, n_W, n_C))
  
  for (i in 1:m) {                        # loop over the batch of training examples
    for (h in 1:n_H) {                    # loop over vertical axis of the output volume
      for (w in 1:n_W) {                  # loop over horizontal axis of the output volume
        for (c in 1:n_C) {                # loop over channels of the output volume
          
          # Find the corners of the current "slice"
          vert_start <- h
          vert_end <- h + f - 1
          horiz_start <- w
          horiz_end <- w + f - 1
          
          # Use the corners to define the current slice on the ith training example of A_prev, channel c.
          a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
          
          # Compute the pooling operation on the slice. Use an if statment to differentiate the modes.
          A[i, h, w, c] <- ifelse(mode == "max", max(a_prev_slice), mean(a_prev_slice))
        }
      }
    }
  }
  
  # Store the input and hparameters in "cache" for pool_backward()
  cache = list(A_prev=A_prev, hparameters=hparameters)
  
  list(A=A, cache=cache)
}


set.seed(1)
A_prev <- array(rnorm(2*4*4*3),c(2,4,4,3))
hparameters <- list(stride=1,f=4)
A <- pool_forward(A_prev, hparameters)$A
cache <- pool_forward(A_prev, hparameters)$cache