require(mxnet)

train <- read.csv('data/train.csv', header=TRUE)
test <- read.csv('data/test.csv', header=TRUE)
train <- data.matrix(train)
test <- data.matrix(test)

train.x <- train[,-1]
train.y <- train[,1]


train.x <- t(train.x/255)
test <- t(test/255)

train.array <- train.x
dim(train.array) <- c(28, 28, 1, ncol(train.x))
test.array <- test
dim(test.array) <- c(28, 28, 1, ncol(test))

# input
data <- mx.symbol.Variable('data')
# first conv
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=6)
relu1 <- mx.symbol.Activation(data=conv1, act_type="relu")
pool1 <- mx.symbol.Pooling(data=relu1, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# second conv
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=16)
relu2 <- mx.symbol.Activation(data=conv2, act_type="relu")
pool2 <- mx.symbol.Pooling(data=relu2, pool_type="max",
                           kernel=c(2,2), stride=c(2,2))
# first fullc
flatten <- mx.symbol.Flatten(data=pool2)
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=84)
relu3 <- mx.symbol.Activation(data=fc1, act_type="relu")
# second fullc
fc2 <- mx.symbol.FullyConnected(data=relu3, num_hidden=10)
# loss
lenet <- mx.symbol.SoftmaxOutput(data=fc2)

# # input
# data <- mx.symbol.Variable('data')
# # first conv
# conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5), num_filter=20)
# tanh1 <- mx.symbol.Activation(data=conv1, act_type="tanh")
# pool1 <- mx.symbol.Pooling(data=tanh1, pool_type="max",
#                            kernel=c(2,2), stride=c(2,2))
# # second conv
# conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5), num_filter=50)
# tanh2 <- mx.symbol.Activation(data=conv2, act_type="tanh")
# pool2 <- mx.symbol.Pooling(data=tanh2, pool_type="max",
#                            kernel=c(2,2), stride=c(2,2))
# # first fullc
# flatten <- mx.symbol.Flatten(data=pool2)
# fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
# tanh3 <- mx.symbol.Activation(data=fc1, act_type="tanh")
# # second fullc
# fc2 <- mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
# # loss
# lenet <- mx.symbol.SoftmaxOutput(data=fc2)

n.gpu <- 1
device.cpu <- mx.cpu()
device.gpu <- lapply(0:(n.gpu-1), function(i) {
  mx.gpu(i)
})

mx.set.seed(0)
tic <- proc.time()
model <- mx.model.FeedForward.create(lenet, X=train.array, y=train.y,
                                     ctx=device.cpu, num.round=12, array.batch.size=128,
                                     learning.rate=0.05, momentum=0.9, wd=0.00001,
                                     eval.metric=mx.metric.accuracy,
                                     epoch.end.callback=mx.callback.log.train.metric(100))

print(proc.time() - tic)

preds <- predict(model, test.array)
pred.label <- max.col(t(preds)) - 1
