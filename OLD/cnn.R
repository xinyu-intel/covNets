rm(list=ls())
sources <- c("mnist.R","rotate.R","convolution.R","relu.R",
             "meanpool.R","model.R","predict.R")

for (i in 1:length(sources)) {
  cat(paste("Loading ",sources[i],"\n"))
  source(sources[i])
}

# load MNIST dataset
provideMNIST(folder = "data/", download = T)
load("data/train.RData")
load("data/test.RData")
dim(trainData) # 60000 784
dim(trainLabels) # 60000 10
dim(testData) # 10000 784
dim(testLabels) # 10000 10

# matrix to tensor
Train_num <- nrow(trainData)
Test_num <- nrow(testData)
X_size <- sqrt(ncol(trainData))
X_train <- array(0,c(Train_num,X_size,X_size))
X_test <- array(0,c(Test_num,X_size,X_size))
for (i in 1:Train_num) {
  X_train[i,,]<-matrix(trainData[i,],X_size)
}
for (i in 1:Test_num) {
  X_test[i,,]<-matrix(testData[i,],X_size)
}

# matrix to vector
Y_train <- max.col(trainLabels)
Y_test <- max.col(testLabels)

samp_train <- 1:Train_num
samp_test <- 1:Test_num

# 2. train model
mnist.model <- model.cnn(X_train = X_train[samp_train,,], 
                      X_test = X_test[samp_test,,], 
                      Y_train = Y_train[samp_train],
                      Y_test = Y_test[samp_test],
                      lr = 1e-2,
                      # regularization rate
                      reg = 1e-3,
                      maxit=2000, display=5)

# 3. prediction
# NOTE: if the predict is factor, we need to transfer the number into class manually.
#       To make the code clear, I don't write this change into predict.dnn function.
labels.cnn <- predict.cnn(mnist.model, data = X_test)

# 4. verify the results
table(Y_test, labels.dnn)

#accuracy
mean(as.integer(Y_test) == labels.dnn)