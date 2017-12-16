rm(list=ls())
sources <- c("mnist.R","pad3d.R","conv_single_step.R","conv_forward.R","conv_backward.R",
             "create_mask_from_window.R","distribute_value.R","pool_forward.R","pool_backward.R")

for (i in 1:length(sources)) {
  cat(paste("Loading ",sources[i],"\n"))
  source(sources[i])
}

provideMNIST(folder = "../data/", download = T)
load("../data/train.RData")
load("../data/test.RData")

dim(trainData) # 60000 784
dim(trainLabels) # 60000 10
dim(testData) # 10000 784
dim(testLabels) # 10000 10

# matrix to tensor
Train_num <- nrow(trainData)
Test_num <- nrow(testData)
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
Y_train <- max.col(trainLabels)
Y_test <- max.col(testLabels)

