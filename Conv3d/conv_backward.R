conv_backward <- function(dZ, cache){
  
  # Implement the backward propagation for a convolution function
  # 
  # Arguments:
  # dZ -- gradient of the cost with respect to the output of the conv layer (Z), array of shape (m, n_H, n_W, n_C)
  # cache -- cache of values needed for the conv_backward(), output of conv_forward()
  # 
  # Returns:
  # dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
  # array of shape (m, n_H_prev, n_W_prev, n_C_prev)
  # dW -- gradient of the cost with respect to the weights of the conv layer (W)
  # array of shape (f, f, n_C_prev, n_C)
  # db -- gradient of the cost with respect to the biases of the conv layer (b)
  # array of shape (1, 1, 1, n_C)
  
  # Retrieve information from "cache"
  A_prev <- cache$A_prev
  W <- cache$W
  b <- cache$b
  hparameters <- cache$hparameters
  
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
  
  # Retrieve dimensions from dZ's shape
  m <- dim(dZ)[1]
  n_H <- dim(dZ)[2]
  n_W <- dim(dZ)[3]
  n_C <- dim(dZ)[4]
  
  # Initialize dA_prev, dW, db with the correct shapes
  dA_prev <- array(0, c(m, n_H_prev, n_W_prev, n_C_prev))
  dW <- array(0, c(f, f, n_C_prev, n_C))
  db <- array(0, c(1, 1, 1, n_C))
  
  # Pad A_prev and dA_prev
  A_prev_pad <- pad3d(A_prev, pad)
  dA_prev_pad <- pad3d(dA_prev, pad)
  
  for (i in 1:m) {                                 # loop over the training examples
    
    # select ith training example from A_prev_pad and dA_prev_pad
    a_prev_pad <- A_prev_pad[i,,,]
    da_prev_pad <- dA_prev_pad[i,,,]
    
    for (h in 1:n_H) {                             # loop over vertical axis of the output volume
      for (w in 1:n_W) {                           # loop over horizontal axis of the output volume
        for (c in 1:n_C) {                         # loop over the channels of the output volume
          
          # Find the corners of the current "slice"
          vert_start <- (h - 1) * stride + 1
          vert_end <- vert_start + f - 1
          horiz_start <- (w - 1) * stride + 1
          horiz_end <- horiz_start + f - 1
          
          # Use the corners to define the slice from a_prev_pad
          a_slice <- a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, ]
          
          # Update gradients for the window and the filter's parameters using the code formulas given above
          da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, ] <- da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, ] + W[,,,c] * dZ[i, h, w, c]
          dW[,,,c] <- dW[,,,c] + a_slice * dZ[i, h, w, c]
          db[,,,c] <- db[,,,c] + dZ[i, h, w, c]
        }
      }
      
    }
    
    # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
    dA_prev[i, , , ] <- da_prev_pad[(pad+1):(pad+n_H_prev), (pad+1):(pad+n_W_prev), ]
  }
  
  list(dA_prev = dA_prev, dW = dW, db = db)
}


# set.seed(1)
# dA <- conv_backward(Z, cache_conv)$dA_prev
# dW <- conv_backward(Z, cache_conv)$dW
# db <- conv_backward(Z, cache_conv)$db