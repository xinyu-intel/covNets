conv_forward <- function(A_prev, W, b, hparameters){
  
  # Implements the forward propagation for a convolution function
  # 
  # Arguments:
  # A_prev -- output activations of the previous layer, array of shape (m, n_H_prev, n_W_prev, n_C_prev)
  # W -- Weights, array of shape (f, f, n_C_prev, n_C)
  # b -- Biases, array of shape (1, 1, 1, n_C)
  # hparameters -- R list containing "stride" and "pad"
  # 
  # Returns:
  # Z -- conv output, array of shape (m, n_H, n_W, n_C)
  # cache -- cache of values needed for the conv_backward() function
  
  # Retrieve dimensions from A_prev's shape
  m <- dim(A_prev)[1]
  n_H_prev <- dim(A_prev)[2]
  n_W_prev <- dim(A_prev)[3]
  n_C_prev <- dim(A_prev)[4]
  
  # Retrieve dimensions from W's shape
  f <- dim(W)[1]
  n_C_prev <- dim(W)[3]
  n_C <- dim(W)[4]
  
  # Retrieve information from "hparameters"
  stride <- hparameters$stride
  pad <- hparameters$pad
  
  # Compute the dimensions of the CONV output volume using the formula given above.
  n_H = floor((n_H_prev-f+2*pad)/stride) + 1
  n_W = floor((n_W_prev-f+2*pad)/stride) + 1
  
  # Initialize the output volume Z with zeros.
  Z <- array(0,c(m, n_H, n_W, n_C))
  
  # Create A_prev_pad by padding A_prev
  A_prev_pad <- pad3d(A_prev, pad)
  
  for (i in 1:m) {                        # loop over the batch of training examples
    a_prev_pad <- A_prev_pad[i,,,]        # Select ith training example's padded activation
    for (h in 1:n_H) {                    # loop over vertical axis of the output volume
      for (w in 1:n_W) {                  # loop over horizontal axis of the output volume
        for (c in 1:n_C) {                # loop over channels (= #filters) of the output volume
          
          # Find the corners of the current "slice"
          vert_start <- h
          vert_end <- h + f - 1
          horiz_start <- w
          horiz_end <- w + f - 1
          
          # Use the corners to define the (3D) slice of a_prev_pad
          a_slice_prev <- a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,]
          
          # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron.
          Z[i, h, w, c] = conv_single_step(a_slice_prev, W[,,,c], b[,,,c])
        }
        
      }
      
    }
    
  }
  
  # Save information in "cache" for the backprop
  cache <- list(A_prev=A_prev, W=W, b=b, hparameters=hparameters)
  
  list(Z=Z,cache=cache)
}


set.seed(1)
A_prev <-  array(rnorm(10*4*4*3), c(10,4,4,3))
W <- array(rnorm(2*2*3*8), c(2,2,3,8))
b <- array(rnorm(8),c(1,1,1,8))
hparameters <- list(pad = 2, stride = 1)

Z <- conv_forward(A_prev, W, b, hparameters)$Z
cache_conv <- conv_forward(A_prev, W, b, hparameters)$cache