# predict
predict.cnn <- function(model, data = X_test) {
  
  # number of testdata
  N_t <- dim(data)[1]
  # init layer
  conv.layer_t <- array(0 ,c(N,C,C))
  pooling.layer_t <- array(0 ,c(N,P,P))
  stretch_t <- matrix(0, N, P*P)
  
  # Feed Forwad
  for (i in 1:N_t) {
    conv.layer_t[i,,] <- convolution(data[i,,], W = model$W1, b = model$b1)
    # neurons : Rectified Linear
    conv.layer_t[i,,] <- relu(conv.layer_t[i,,])
    # pooling layer
    pooling.layer_t[i,,] <- meanpool(conv.layer_t[i,,], ksize = pooling_size, stride = pooling_stride)
    # tensor to matrix
    stretch_t[i,] <- matrix(pooling.layer_t[i,,], ncol = 1)
  }
  
  # affine layer
  score_t <- sweep(stretch_t %*% model$W2, 2, model$b2, '+')
  
  # softmax
  score.exp_t <- exp(score_t)
  probs_t <-sweep(score.exp_t, 1, rowSums(score.exp_t), '/') 
  
  # select max possiblity
  labels.predicted <- max.col(probs_T)
  return(labels.predicted)
}