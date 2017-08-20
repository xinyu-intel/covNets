cnnPredict <- function(model, data = X.test, C, P) {
  
  conv.layer_test <- array(0, c(dim(data)[1],C,C))
  pooling.layer_test <- array(0, c(dim(data)[1],P,P))
  shaping_test <- matrix(0,dim(data)[1],P*P)
  
  # Feed Forwad
  for (i in 1:dim(data)[1]) {
    conv.layer_test[i,,] <- convolution(input_img = data[i,,], W = model$W1, b = model$b1)
    # neurons : Rectified Linear
    conv.layer_test[i,,] <- relu(conv.layer_test[i,,])
    # pooling layer
    pooling.layer_test[i,,] <- meanpool(conv.layer_test[i,,], 2, 2)
    shaping_test[i,] <- matrix(pooling.layer_test[i,,], nrow = 1)
  }
  # affine layer
  score_test <- sweep(shaping_test %*% model$W2, 2, model$b2, '+')
  
  # Loss Function: softmax
  score.exp_test <- exp(score_test)
  probs_test <-sweep(score.exp_test, 1, rowSums(score.exp_test), '/') 
  
  # select max possiblity
  labels.predicted <- max.col(probs_test)
  labels.predicted
}