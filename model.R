rm(list=ls())
sources <- c("mnist.R","pad3d.R","conv_single_step.R","conv_forward.R","conv_backward.R",
             "create_mask_from_window.R","distribute_value.R","pool_forward.R","pool_backward.R",
             "arr2col.R", "affine_forward.R","activation.R","affine_activation_forward.R",
             "softmax.R","cross_entropy_cost.R","affine_backward.R","affine_activation_backward.R")

for (i in 1:length(sources)) {
  cat(paste("Loading ",sources[i],"\n"))
  source(sources[i])
}

provideMNIST(folder = "./data/", download = T)
load("./data/train.RData")
load("./data/test.RData")

dim(trainData) # 60000 784
dim(trainLabels) # 60000 10
dim(testData) # 10000 784
dim(testLabels) # 10000 10

# matrix to tensor
# Train_num <- nrow(trainData)
# Test_num <- nrow(testData)

batch_size <- 300
Train_num <- batch_size
Test_num <- batch_size
X_size <- sqrt(ncol(trainData))
X_train <- array(0,c(Train_num,X_size,X_size,1))
X_test <- array(0,c(Test_num,X_size,X_size,1))
for (i in 1:Train_num) {
  X_train[i,,,]<-matrix(trainData[i,],X_size,1)
}
for (i in 1:Test_num) {
  X_test[i,,,]<-matrix(testData[i,],X_size,1)
}

# matrix to vector
Y_train <- max.col(trainLabels)[1:batch_size]
Y_test <- max.col(testLabels)[1:batch_size]
Y.len   <- length(unique(Y_train))

set.seed(1)

# iteration
iter <- 0
maxit <- 50
while(iter < maxit){
  
  iter <- iter + 1
  
  # weight initialization
  if (!exists('W')) {
    f <- 3
    n_C <- 3
    W <- array(rnorm(f*f*dim(X_train)[4]*n_C), c(f, f, dim(X_train)[4], n_C))
    b <- array(0,c(1, 1, 1, n_C))
  }
  
  # conv forward
  hparameters <- list(pad = 1, stride = 1)
  conv_forward_output <- conv_forward(X_train, W, b, hparameters)
  Z <- conv_forward_output$Z
  cache_conv <- conv_forward_output$cache
  
  # conv relu
  relu_output <- relu(Z)
  Z <- relu_output$A
  cache_relu <- relu_output$cache
  
  # pool forward
  pool_forward_output <- pool_forward(Z, list(stride=4,f=4), mode = "max")
  P <- pool_forward_output$A
  cache_pool <- pool_forward_output$cache
  
  # stretch
  stretch_output_list <- arr2col(P)
  stretch_output <- stretch_output_list$output
  cache_stretch <- stretch_output_list$cache
  
  if (!exists('W2')) {
    # affine layer initialization
    # He Initialization
    W2 <- matrix(rnorm(Y.len * dim(stretch_output)[2]),
                 nrow=dim(stretch_output)[2], ncol=Y.len) * sqrt(2/dim(stretch_output)[2])
    b2 <- matrix(0, nrow=1, ncol=Y.len)
  }
  
  # affine forward
  affine_activation_output <- affine_activation_forward(stretch_output, W2, b2, activation = "NULL")
  A <- affine_activation_output$A
  cache_affine <- affine_activation_output$cache
  
  # softmax layer
  softmax_output <- softmax(A)
  S <- softmax_output$probs
  cache_softmax <- softmax_output$cache
  pred <- max.col(S)
  
  # loss function
  loss_output <- cross_entropy_cost(AL = S, Y = Y_train, batch_size = batch_size)
  loss <- loss_output$loss
  cache_loss <- loss_output$cache
  accuracy <- mean(as.integer(Y_train) == pred)
  
  # compute dA
  AL <- cache_loss$AL
  Y.index <- cache_loss$Y.index
  dA <- AL
  dA[Y.index] <- dA[Y.index] - 1
  dA <- dA/batch_size
  
  # affine backward
  affine_activation_backward_output <- affine_activation_backward(dA, cache = cache_affine, activation = "NULL")
  daffine <- affine_activation_backward_output$dA_prev
  dW2 <- affine_activation_backward_output$dW
  db2 <- affine_activation_backward_output$db
  
  # unstretch
  unstrentch <- col2arr(daffine, cache = cache_stretch)
  
  # pooling backward
  pool_backward_output <- pool_backward(unstrentch, cache = cache_pool, mode = "max")
  
  # relu backward
  pool_backward_output <- relu_backward(pool_backward_output, cache = cache_relu)
  
  # conv backward
  conv_backward_output <- conv_backward(pool_backward_output, cache = cache_conv)
  dconv <- conv_backward_output$dA_prev
  dW <- conv_backward_output$dW
  db <- conv_backward_output$db
  
  # update weights
  lr <- 1e-1
  
  W <- W - lr * dW
  b <- b - lr * db
  
  W2 <- W2 - lr * dW2
  b2 <- b2 - lr * db2
  
  cat(iter, loss, accuracy, "\n")
  
}
