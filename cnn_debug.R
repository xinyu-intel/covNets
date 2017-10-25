load("mnist.RData")

# rotate matrix
rotate <- function(x) t(apply(x, 2, rev))

# convolution layer
convolution <- function(input_img, W, b, show=FALSE, out=TRUE)
{
  kernel_size <- dim(W)[1]
  conv_out <- outer(
    1:(nrow(input_img)-kernel_size+1),
    1:(ncol(input_img)-kernel_size+1),
    Vectorize(function(r,c) sum(input_img[r:(r+kernel_size-1),
                                          c:(c+kernel_size-1)]*W)+b)
  )    
  if (show){imgshow(conv_out)}
  if (out){conv_out}
}

# meanpool layer
meanpool <- function(input_img, ksize, stride, show=FALSE, out=TRUE)
{
  conv_out <- outer(
    1:(((nrow(input_img)-ksize)/stride)+1),
    1:(((ncol(input_img)-ksize)/stride)+1),
    Vectorize(function(r,c) mean(input_img[
      ((r*stride)-(stride-1)):(((r*stride)-(stride))+ksize),
      ((c*stride)-(stride-1)):(((c*stride)-(stride))+ksize)]))
  )    
  if (show){imgshow(conv_out)}
  if (out){conv_out}
}

# relu
relu <- function(z){z*(z>0)}

set.seed(1)

Y_train[Y_train==0]<-10
Y_test[Y_test==0]<-10

# 1. split data into test/train
samp_train <- sample(1:60000,300)
samp_test <- sample(1:10000,50)

X_train = X_train[samp_train,,]
X_test = X_test[samp_test,,]
Y_train = Y_train[samp_train]
Y_test = Y_test[samp_test]

# predict
predict.cnn <- function(model, data = X.test) {
  
  conv.layer_test <- array(0, c(dim(data)[1],26,26))
  pooling.layer_test <- array(0, c(dim(data)[1],13,13))
  shaping_test <- matrix(0,dim(data)[1],13*13)
  
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
  score <- sweep(shaping_test %*% model$W2, 2, model$b2, '+')
  
  # Loss Function: softmax
  score.exp <- exp(score)
  probs <-sweep(score.exp, 1, rowSums(score.exp), '/') 
  
  # select max possiblity
  labels.predicted <- max.col(probs)
  return(labels.predicted)
}


# training
maxit=2000
display=50

# total number of training set
N <- dim(X_train)[1]

# extract the data and label
# don't need atribute 
X <- X_train
# correct categories represented by integer 
Y <- Y_train
if(is.factor(Y)) { Y <- as.integer(Y) }
# create index for both row and col
# create index for both row and col
Y.len <- length(unique(Y))
Y.set <- sort(unique(Y))
Y.index <- cbind(1:N, match(Y, Y.set))


model <- NULL
# create model or get model from parameter
if(is.null(model)) {
  # size of filter
  F <- 3
  # size of conv_layer
  conv_layer_size <- dim(X)[2]-F+1
  # size of pooling layer
  pooling_layer_size <- conv_layer_size/2
  # number of categories for classification
  K <- length(unique(Y))
  
  # create and init weights and bias 
  W1 <- matrix(rnorm(F*F), nrow=F)/sqrt(F*F)
  b1 <- rnorm(1)
  
  W2 <- matrix(rnorm(pooling_layer_size*pooling_layer_size*K),
               nrow=pooling_layer_size*pooling_layer_size,
               ncol=K)/sqrt(pooling_layer_size*pooling_layer_size*K)
  b2 <- matrix(0, nrow=1, ncol=K)
} else {
  K  <- model$K
  W1 <- model$W1
  b1 <- model$b1
  W2 <- model$W2
  b2 <- model$b2
}

# use all train data to update weights since it's a small dataset
batchsize <- N
# init loss to a very big value
loss <- 100000

conv.layer <- array(0,c(N,conv_layer_size,conv_layer_size))
pooling.layer <- array(0,c(N,pooling_layer_size,pooling_layer_size))
shaping <- matrix(0,N,pooling_layer_size*pooling_layer_size)
score <- matrix(0, N, Y.len)
score.exp <- matrix(0, N, Y.len)
probs <- matrix(0, N, Y.len)

t <- 0
while(t < maxit) {
  
  # iteration index
  t <- t +1
# Training the network
  lr <- 1e-1
  # forward ....
  # 1 indicate row, 2 indicate col
  corect.logprobs <- 0
  for (i in 1:N) {
    # forward ....
    # 1 indicate row, 2 indicate col
    conv.layer[i,,] <- convolution(X_train[i,,], W = W1, b = b1)
    # neurons : ReLU
    conv.layer[i,,] <- relu(conv.layer[i,,])
    # pooling layer
    pooling.layer[i,,] <- meanpool(conv.layer[i,,], 2, 2)
    shaping[i,] <- matrix(pooling.layer[i,,], nrow = 1)
  }
  
  # affine layer
  score <- sweep(shaping %*% W2, 2, b2, '+')
  # softmax
  score.exp <- exp(score)
  # debug
  probs <- score.exp/rowSums(score.exp)
  # compute the loss
  corect.logprobs <- -log(probs[Y.index])
  
  loss <- sum(corect.logprobs)/batchsize
  
  # display results and update model
  if( t %% display == 0) {
    if(!is.null(X_test)) {
      model <- list(K = K,
                     # weights and bias
                     W1 = W1, 
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
  dscores <- dscores / batchsize
  
  dW2 <- t(shaping) %*% dscores 
  db2 <- colSums(dscores)
  
  dshaping <- dscores %*% t(W2)
  shape_error <- matrix(0,pooling_layer_size,pooling_layer_size)
  unshaping <- array(0,c(N,pooling_layer_size,pooling_layer_size))
  for (i in i:N) {
    unshaping[i,,] <- matrix(dshaping[i,],13,13)
    shape_error <- shape_error + unshaping[i,,]
  }
  
  upsample <- matrix(0,conv_layer_size,conv_layer_size)
    for (i in 1:pooling_layer_size) {
      for (j in 1:pooling_layer_size) {
        upsample[2*i-1,2*j-1] <- shape_error[i,j]
        upsample[2*i-1,2*j] <- shape_error[i,j]
        upsample[2*i,2*j-1] <- shape_error[i,j]
        upsample[2*i,2*j] <- shape_error[i,j]
      }
    }
  upsample <- upsample/4
  upsample[upsample <= 0] <- 0
  dconv <- upsample
  
  dW1 <- matrix(0,F,F)
  for (i in i:N) {
    dW1 <- dW1 + convolution(X_train[i,,], W = rotate(dconv), b = 0)
  }

  dW1 <- dW1 / N

  db1 <- sum(dconv)
  
  # update ....

  
  W1 <- W1 - lr * dW1
  b1 <- b1 - lr * db1
  
  W2 <- W2 - lr * dW2
  b2 <- b2 - lr * db2
}

  model <- list( K = K,
                 # weights and bias
                 W1= W1, 
                 b1= b1, 
                 W2= W2, 
                 b2= b2)
  return(model)
  
