# Train: build and train a 2-layers neural network 
model.cnn <- function(X_train = NULL, X_test = NULL, 
                      Y_train = NULL, Y_test = NULL,
                      model = NULL,
                      # set filter size
                      filter_size = 3,
                      # set filter stride
                      filter_stride = 1,
                      # set pooling size
                      pooling_size = 2,
                      # set pooling stride
                      pooling_stride = 2,
                      # max iteration steps
                      maxit=2000,
                      # delta loss 
                      abstol=1e-2,
                      # learning rate
                      lr = 1e-2,
                      # regularization rate
                      reg = 1e-3,
                      # show results every 'display' step
                      display = 100,
                      random.seed = 1)
{
  # to make the case reproducible.
  set.seed(random.seed)
  
  # total number of training set
  N <- dim(X_train)[1]
  
  X <- X_train
  # correct categories represented by integer
  Y <- Y_train
  if(is.factor(Y)) { Y <- as.integer(Y) }
  # create index for both row and col
  Y.len <- length(unique(Y))
  Y.set <- sort(unique(Y))
  Y.index <- cbind(1:N, match(Y, Y.set))
  
  # create model or get model from parameter
  if(is.null(model)) {
    # size of filter
    F <- filter_size
    # size of conv_layer
    C <- dim(X)[2]-F+filter_stride
    # size of pooling layer
    P <- C/pooling_stride
    # number of categories for classification
    K <- Y.len
    
    # create and init weights and bias 
    W1 <- matrix(rnorm(F*F), nrow=F)
    b1 <- rnorm(1)
    # He Initialization
    W2 <- matrix(rnorm(P*P*K),
                 nrow=P*P,ncol=K)*sqrt(2/P)
    b2 <- matrix(0, nrow=1, ncol=K)
    
  } else {
    W1 <- model$W1
    b1 <- model$b1
    W2 <- model$W2
    b2 <- model$b2
  }
  
  # init loss to a very big value
  loss <- 100000
  
  # init layer
  conv.layer <- array(0,c(N,C,C))
  pooling.layer <- array(0,c(N,P,P))
  stretch <- matrix(0, N, P*P)
  
  # Training the network
  t <- 0
  while(t < maxit) {
    
    # iteration index
    t <- t +1
    
    # forward ....
    # 1 indicate row, 2 indicate col
    corect.logprobs <- 0
    
    # conv block(conv+relu+pooling)
    for (i in 1:N) {
       # forward ....
       # 1 indicate row, 2 indicate col
       conv.layer[i,,] <- convolution(X_train[i,,], W = W1, b = b1)
       # neurons : ReLU
       conv.layer[i,,] <- relu(conv.layer[i,,])
       # pooling layer
       pooling.layer[i,,] <- meanpool(conv.layer[i,,], ksize = pooling_size, stride = pooling_stride)
       # tensor to matrix
       stretch[i,] <- matrix(pooling.layer[i,,], ncol = 1)
    }
    
    # affine layer
    score <- sweep(stretch %*% W2, 2, b2, '+')
    # softmax
    score.exp <- exp(score)
    probs <- score.exp/rowSums(score.exp)
    # compute the loss
    corect.logprobs <- -log(probs[Y.index])
    data.loss <- sum(corect.logprobs)/N
    reg.loss <- 0.5*reg* (sum(W1*W1) + sum(W2*W2))
    loss <- data.loss + reg.loss
    
    # display results and update model
    if( t %% display == 0) {
      if(!is.null(X_test)) {
        model <- list(W1 = W1, 
                       b1 = b1, 
                       W2 = W2, 
                       b2 = b2)
        labs <- predict.cnn(model, X_test)
        accuracy <- mean(as.integer(Y_test) == Y.set[labs])
        cat(t, loss, accuracy, "\n")
      } else {
        cat(t, loss, "\n")
      }
    }
    
  # backward ....
  dscores <- probs
  dscores[Y.index] <- dscores[Y.index] -1
  
  dW2 <- t(stretch) %*% dscores 
  db2 <- colSums(dscores)
  
  dstretch <- dscores %*% t(W2)
  unstretch <- array(0,c(N,P,P))
  for (i in i:N) {
    unstretch[i,,] <- matrix(dstretch[i,],P,P)
  }

  upsample <- array(0,c(N,C,C))
  for (n in 1:N) {
    for (i in 1:P) {
      for (j in 1:P) {
        upsample[n,2*i-1,2*j-1] <- unstretch[n,i,j]
        upsample[n,2*i-1,2*j] <- unstretch[n,i,j]
        upsample[n,2*i,2*j-1] <- unstretch[n,i,j]
        upsample[n,2*i,2*j] <- unstretch[n,i,j]
      }
    }
  }
   
  upsample <- upsample/4
  upsample[upsample <= 0] <- 0
  
  ####### blow may be wrong ######
  dconv <- apply(upsample,2:3,sum)
  
  dW1 <- matrix(0,F,F)
  
  for (i in i:N) {
    dW1 <- dW1 + convolution(X_train[i,,], W = rotate(dconv), b = 0)
  }

  dW1 <- dW1 / N

  db1 <- sum(dconv)
  
  # update ....
  dW2 <- dW2 + reg*W2
  dW1 <- dW1 + reg*W1
  
  W1 <- W1 - lr * dW1
  b1 <- b1 - lr * db1
  
  W2 <- W2 - lr * dW2
  b2 <- b2 - lr * db2
  }
  
  # final results
  # creat list to store learned parameters
  # you can add more parameters for debug and visualization
  # such as residuals, fitted.values ...
  model <- list( W1= W1, 
                 b1= b1, 
                 W2= W2, 
                 b2= b2)
  
  return(model)
}