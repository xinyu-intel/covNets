cnnBackpropagation <- function(probs, N, batchsize, conv_layer_size, pooling_layer_size, pooling_size, filter_size){
  # backward ....
  dscores <- probs
  dscores[Y.index] <- dscores[Y.index] -1
  dscores <- dscores / batchsize
  
  dW2 <- t(shaping) %*% dscores 
  db2 <- colSums(dscores)
  
  upsample <- upsample(N, conv_layer_size, pooling_layer_size, pooling_size)
  upsample[upsample <= 0] <- 0
  dconv <- upsample
  
  dconv_sum <- matrix(0,conv_layer_size,conv_layer_size)
  for (i in 1:N) {
    dconv_sum <- dconv[i,,] + dconv_sum
  }
  
  dW1 <- matrix(0,filter_size,filter_size)
  
  for (i in 1:N) {
    dW1 <- dW1 + convolution(rotate(rotate(X_train[i,,])), W = dconv_sum, b = 0)
  }
  
  db1 <- sum(dconv_sum)
  
  # update ....
  dW2 <- dW2 + reg*W2
  dW1 <- dW1 + reg*W1
  
  W1 <- W1 - lr * dW1
  b1 <- b1 - lr * db1
  
  W2 <- W2 - lr * dW2
  b2 <- b2 - lr * db2
  
  model <- list( K = K,
                 C = C,
                 F = F,
                 P = P,
                 # weights and bias
                 W1= W1, 
                 b1= b1, 
                 W2= W2, 
                 b2= b2)
  
  return(model)
}