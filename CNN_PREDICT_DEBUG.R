# cnn predict debug

data <- X_test
conv.layer_test <- array(0, c(dim(data)[1],26,26))
pooling.layer_test <- array(0, c(dim(data)[1],13,13))
shaping_test <- matrix(0,dim(data)[1],13*13)

for (i in 1:dim(data)[1]) {
  conv.layer_test[i,,] <- convolution(input_img = data[i,,], W = model$W1, b = model$b1)
  # neurons : Rectified Linear
  conv.layer_test[i,,] <- relu(conv.layer_test[i,,])
  # pooling layer
  pooling.layer_test[i,,] <- avgpool(conv.layer_test[i,,], 2, 2)
  shaping_test[i,] <- matrix(pooling.layer_test[i,,], nrow = 1)
}
# affine layer
score <- sweep(shaping_test %*% model$W2, 2, model$b2, '+')

# Loss Function: softmax
score.exp <- exp(score)
probs <-sweep(score.exp, 1, rowSums(score.exp), '/') 

# select max possiblity
labels.predicted <- max.col(probs)
returnValue(labels.predicted)


labs <- labels.predicted
accuracy <- mean(as.integer(Y_test) == Y.set[labs])
cat(i, loss, accuracy, "\n")