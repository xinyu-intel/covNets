rm(list=ls())
sources <- c("pad3d.R","conv_single_step.R","conv_forward.R","conv_backward.R",
             "create_mask_from_window.R","distribute_value.R","pool_forward.R","pool_backward.R",
             "arr2col.R", "affine_forward.R","activation.R","affine_activation_forward.R",
             "softmax.R","cross_entropy_cost.R","affine_backward.R","affine_activation_backward.R")

for (i in 1:length(sources)) {
  cat(paste("Loading ",sources[i],"\n"))
  source(sources[i])
}

train <- read.csv('data/train.csv', header=TRUE)
test <- read.csv('data/test.csv', header=TRUE)
train <- data.matrix(train)
test <- data.matrix(test)

train.x <- train[,-1]
test.x <- test

batch_size <- 100

train.x <- t(train.x/255)[,1:batch_size]
train.y <- train[,1][1:batch_size]
test.x <- t(test.x/255)[,1:batch_size]

train.array <- train.x
dim(train.array) <- c(ncol(train.x), 28, 28, 1)
test.array <- test.x
dim(test.array) <- c(ncol(test.x), 28, 28, 1)

Train_num <- batch_size
Test_num <- batch_size

X_size <- sqrt(nrow(train.x))

Y.len <- length(unique(train.y))

set.seed(1)

# iteration
iter <- 0
maxit <- 5000
while(iter < maxit){
  
  iter <- iter + 1
  
  # ============ conv block 0 ============
  # weight initialization
  if (!exists('W')) {
    f <- 5
    n_C <- 6
    W <- array(rnorm(f*f*dim(train.array)[4]*n_C), c(f, f, dim(train.array)[4], n_C))
    b <- array(0,c(1, 1, 1, n_C))
  }
  
  # conv forward (filter: 5*5*6)
  hparameters <- list(pad = 2, stride = 1)
  conv_forward_output <- conv_forward(train.array, W, b, hparameters)
  Z <- conv_forward_output$Z
  cache_conv <- conv_forward_output$cache
  
  # conv relu
  relu_output <- relu(Z)
  Z <- relu_output$A
  cache_relu <- relu_output$cache
  
  # pool forward
  pool_forward_output <- pool_forward(Z, list(stride=2,f=2), mode = "max")
  P <- pool_forward_output$A
  cache_pool <- pool_forward_output$cache
  
  # ============ conv block 1 ============
  # weight initialization
  if (!exists('W1')) {
    f1 <- 5
    n_C1 <- 16
    W1 <- array(rnorm(f1*f1*dim(P)[4]*n_C1), c(f1, f1, dim(P)[4], n_C1))
    b1 <- array(0,c(1, 1, 1, n_C1))
  }
  
  # conv forward (filter: 5*5*16)
  hparameters1 <- list(pad = 0, stride = 1)
  conv_forward_output1 <- conv_forward(P, W1, b1, hparameters1)
  Z1 <- conv_forward_output1$Z
  cache_conv1 <- conv_forward_output1$cache
  
  # conv relu
  relu_output1 <- relu(Z1)
  Z1 <- relu_output1$A
  cache_relu1 <- relu_output1$cache
  
  # pool forward
  pool_forward_output1 <- pool_forward(Z1, list(stride=2,f=2), mode = "max")
  P1 <- pool_forward_output1$A
  cache_pool1 <- pool_forward_output1$cache
  
  # ============ conv block 2 ============
  # weight initialization
  if (!exists('W2')) {
    f2 <- 5
    n_C2 <- 120
    W2 <- array(rnorm(f2*f2*dim(P1)[4]*n_C2), c(f2, f2, dim(P1)[4], n_C2))
    b2 <- array(0,c(1, 1, 1, n_C2))
  }
  
  # conv forward (filter: 5*5*120)
  hparameters2 <- list(pad = 0, stride = 1)
  conv_forward_output2 <- conv_forward(P1, W2, b2, hparameters2)
  Z2 <- conv_forward_output2$Z
  cache_conv2 <- conv_forward_output2$cache
  
  stretch_output <- matrix(Z2,dim(Z2)[1],dim(Z2)[4])
  
  # ============ affine layer 0 ============
  
  if (!exists('W3')) {
    # affine layer initialization
    # He Initialization
    W3 <- matrix(rnorm(84 * dim(stretch_output)[2]),
                 nrow=dim(stretch_output)[2], ncol=84) * sqrt(2/dim(stretch_output)[2])
    b3 <- matrix(0, nrow=1, ncol=84)
  }
  
  # affine forward (84)
  affine_activation_output0 <- affine_activation_forward(stretch_output, W3, b3, activation = "relu")
  A0 <- affine_activation_output0$A
  cache_affine0 <- affine_activation_output0$cache
  
  # ============ affine layer 1 ============
  
  if (!exists('W4')) {
    # affine layer initialization
    # He Initialization
    W4 <- matrix(rnorm(Y.len * dim(A0)[2]),
                 nrow=dim(A0)[2], ncol=Y.len) * sqrt(2/dim(A0)[2])
    b4 <- matrix(0, nrow=1, ncol=Y.len)
  }
  
  # affine forward (10)
  affine_activation_output1 <- affine_activation_forward(A0, W4, b4, activation = "NULL")
  A1 <- affine_activation_output1$A
  cache_affine1 <- affine_activation_output1$cache
  
  # =========== softmax and loss function ============
  # softmax layer
  softmax_output <- softmax(A1)
  S <- softmax_output$probs
  cache_softmax <- softmax_output$cache
  pred <- max.col(S)
  
  # loss function
  loss_output <- cross_entropy_cost(AL = S, Y = train.y, batch_size = batch_size)
  loss <- loss_output$loss
  cache_loss <- loss_output$cache
  accuracy <- mean(as.integer(train.y) == pred)
  
  # ============ backpropgation ============
  # compute dA1
  AL <- cache_loss$AL
  Y.index <- cache_loss$Y.index
  dA1 <- AL
  dA1[Y.index] <- dA1[Y.index] - 1
  dA1 <- dA1/batch_size
  
  # affine backward 1
  affine_activation_backward_output1 <- affine_activation_backward(dA1, cache = cache_affine1, activation = "NULL")
  daffine1 <- affine_activation_backward_output1$dA_prev
  dW4 <- affine_activation_backward_output1$dW
  db4 <- affine_activation_backward_output1$db
  
  # affine backward 0
  
  affine_activation_backward_output0 <- affine_activation_backward(daffine1, cache = cache_affine0, activation = "relu")
  daffine0 <- affine_activation_backward_output0$dA_prev
  dW3 <- affine_activation_backward_output0$dW
  db3 <- affine_activation_backward_output0$db
  
  # ============ conv block back 2 ============
  # unstretch
  unstrentch <- array(daffine0, c(dim(daffine0)[1],1,1,dim(daffine0)[2]))
  
  # conv backward 2
  conv_backward_output2 <- conv_backward(unstrentch, cache = cache_conv2)
  dconv2 <- conv_backward_output2$dA_prev
  dW2 <- conv_backward_output2$dW
  db2 <- conv_backward_output2$db
  
  # ============ conv block back 1 ============
  # pooling backward
  pool_backward_output1 <- pool_backward(dconv2, cache = cache_pool1, mode = "max")
  
  # relu backward
  pool_backward_output1 <- relu_backward(pool_backward_output1, cache = cache_relu1)
  
  # conv backward
  conv_backward_output1 <- conv_backward(pool_backward_output1, cache = cache_conv1)
  dconv1 <- conv_backward_output1$dA_prev
  dW1 <- conv_backward_output1$dW
  db1 <- conv_backward_output1$db
  
  # ============ conv block back 0 ============
  # pooling backward
  pool_backward_output <- pool_backward(dconv1, cache = cache_pool, mode = "max")
  
  # relu backward
  pool_backward_output <- relu_backward(pool_backward_output, cache = cache_relu)
  
  # conv backward
  conv_backward_output <- conv_backward(pool_backward_output, cache = cache_conv)
  dconv <- conv_backward_output$dA_prev
  dW <- conv_backward_output$dW
  db <- conv_backward_output$db
  
  # ============ update weights ============
  lr <- 0.06
  
  W <- W - lr * dW
  b <- b - lr * db
  
  W1 <- W1 - lr * dW1
  b1 <- b1 - lr * db1
  
  W2 <- W2 - lr * dW2
  b2 <- b2 - lr * db2
  
  W3 <- W3 - lr * dW3
  b3 <- b3 - lr * db3
  
  W4 <- W4 - lr * dW4
  b4 <- b4 - lr * db4
  
 # if( iter %% 100 == 0){
    cat(iter, loss, accuracy, "\n")
 # }
  
  
}
