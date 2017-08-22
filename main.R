## ------------------- Part1: Initialization ------------------- 

# Load Data
rm(list=ls())
sources <- c("convolution.R","meanpool.R","relu.R","rotate.R",
             "cnnForward.R","cnnCostFunction.R","cnnBackpropagation.R",
            "cnnPredict.R")
for (i in 1:length(sources)) {
  cat(paste("Loading ",sources[i],"\n"))
  source(sources[i])
}
cat(sprintf('Loading Data ...\n'))
load("mnist.RData")

# to make the case reproducible.
set.seed(1)
# sampling to a small dataset
samp_train <- sample(1:60000,300)
samp_test <- sample(1:10000,50)
X <- X_train[samp_train,,]
Y <- Y_train[samp_train]
X_t <- X_test[samp_test,,]
Y_t <- Y_test[samp_test]
Y[Y==0]<-10
Y_t[Y_t==0]<-10

# total number of training set
N <- dim(X)[1]
# create index for both row and col
# create index for both row and col
Y.len <- length(unique(Y))
Y.set <- sort(unique(Y))
Y.index <- cbind(1:N, match(Y, Y.set))

# Model Initialization
model <- NULL
# create model or get model from parameter
if(is.null(model)) {
  # size of filter
  F <- 3
  # size of conv_layer
  C <- dim(X)[2]-F+1
  # size of pooling layer
  P <- C/2
  # number of categories for classification
  K <- Y.len
  
  # create and init weights and bias 
  W1 <- matrix(rnorm(F*F), nrow=F)/sqrt(F*F)
  b1 <- rnorm(1)
  W2 <- matrix(rnorm(P*P*K),nrow=P*P,ncol=K)/sqrt(P*P*K)
  b2 <- matrix(0, nrow=1, ncol=K)
  
  model <- list( F = F,
                 C = C,
                 P = P,
                 K = K,
                 # weights and bias
                 W1 = W1, 
                 b1 = b1, 
                 W2 = W2, 
                 b2 = b2)
  
} else {
  F  <- model$F
  C  <- model$C
  P  <- model$P
  K  <- model$K
  W1 <- model$W1
  b1 <- model$b1
  W2 <- model$W2
  b2 <- model$b2
}


## ------------------- Part2: Compute Cost (Feedforward) ------------------- 
cat(sprintf('\nFeedforward Using Convolutional Neural Network ...\n'))
reg <- 1e-1
probs <- cnnForward(N, X = X_train, model)
loss <- cnnCostFunction(probs, model, N)
cat(sprintf(('Cost at parameters (initialization): %f'), loss))

## ------------------- Part3: Implement Backpropagation ------------------- 
cat(sprintf('\nBackpropagation... \n'))
lr <- 1e-1
model <- cnnBackpropagation(probs, batchsize=N, N, C, P, pooling_size=2, F)

## ------------------- Part4: training CNN ------------------- 
cat(sprintf('\nTraining Convolutional Neural Network... \n'))
maxit <- 2000
abstol <- 1e-2
display <- 10
lr <- 1e-1
reg <- 1e-3
t <- 0
while(t < maxit && loss > abstol ) {
  
  # iteration index
  t <- t +1
  
  # forward ....
  probs <- cnnForward(N, X = X_train, model)
  # compute the loss
  loss <- cnnCostFunction(probs, model, N)
  
  # display results and update model
  if( t %% display == 0) {
    if(!is.null(X_t)) {
      model <- list( F = F,
                     C = C,
                     P = P,
                     K = K,
                     # weights and bias
                     W1 = W1, 
                     b1 = b1, 
                     W2 = W2, 
                     b2 = b2)
      labs <- cnnPredict(model, X_t, C, P)
      accuracy <- mean(as.integer(Y_t) == Y.set[labs])
      cat(t, loss, accuracy, "\n")
    } else {
      cat(t, loss, "\n")
    }
  }
  
  # backward ....
  model <- cnnBackpropagation(probs, batchsize=N, N, C, P, pooling_size=2, F, model)
}
mnist.model <- model

## ------------------- Part5: prediction ------------------- 
# NOTE: if the predict is factor, we need to transfer the number into class manually.
#       To make the code clear, I don't write this change into predict.dnn function.
labels.cnn <- cnnPredict(mnist.model, data = X_t, C, P)

# verify the results
table(Y_t, labels.cnn)

#accuracy
mean(as.integer(Y_t) == labels.cnn)