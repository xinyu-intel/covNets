load("mnist.RData")

Y_train[Y_train==0]<-10
Y_test[Y_test==0]<-10

samp <- sample(1:60000,150)
X_train = X_train[samp,,]
Y_train = Y_train[samp]
N <- dim(X_train)[1]

Y <- Y_train
Y.len <- length(unique(Y))
Y.set <- sort(unique(Y))
Y.index <- cbind(1:N, match(Y, Y.set))
batchsize <- N

H <- 6
F <- 3
K <- 10
pooling_layer_size <- 13
W1 <- matrix(rnorm(F*F), nrow=F)/sqrt(F*F)
b1 <- rnorm(1)

W2 <- matrix(rnorm(pooling_layer_size*pooling_layer_size*K),
             nrow=pooling_layer_size*pooling_layer_size,
             ncol=K)/sqrt(pooling_layer_size*pooling_layer_size*K)
b2 <- matrix(0, nrow=1, ncol=K)

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

avgpool <- function(input_img, ksize, stride, show=FALSE, out=TRUE)
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

relu <- function(z){z*(z>0)}

rotate <- function(x) t(apply(x, 2, rev))

conv_layer_size <- 26
pooling_layer_size <- 13
conv.layer <- array(0,c(N,conv_layer_size,conv_layer_size))
pooling.layer <- array(0,c(N,pooling_layer_size,pooling_layer_size))
shaping <- matrix(0,N,pooling_layer_size*pooling_layer_size)
score <- matrix(0, N, Y.len)
score.exp <- matrix(0, N, Y.len)
probs <- matrix(0, N, Y.len)

corect.logprobs <- 0
for (i in 1:N) {
  # forward ....
  # 1 indicate row, 2 indicate col
  conv.layer[i,,] <- convolution(X_train[i,,], W = W1, b = b1)
  # neurons : ReLU
  conv.layer[i,,] <- relu(conv.layer[i,,])
  # pooling layer
  pooling.layer[i,,] <- avgpool(conv.layer[i,,], 2, 2)
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
data.loss <- sum(corect.logprobs)/150
reg.loss <- 0.5*1e-1* (sum(W1*W1) + sum(W2*W2))
loss <- data.loss + reg.loss

model <- list( D = D,
               H = H,
               F = F,
               K = K,
               # weights and bias
               W1 = W1, 
               b1 = b1, 
               W2 = W2, 
               b2 = b2)

# backward ....
dscores <- probs
dscores[Y.index] <- dscores[Y.index] -1
dscores <- dscores / batchsize

dW2 <- t(shaping) %*% dscores 
db2 <- colSums(dscores)

## 这边开始就弄不清楚了
upsample <- array(0,c(N,conv_layer_size,conv_layer_size))
for (k in 1:N) {
  for (i in 1:pooling_layer_size) {
    for (j in 1:pooling_layer_size) {
      upsample[k,2*i-1,2*j-1] <- pooling.layer[k,i,j]
      upsample[k,2*i-1,2*j] <- pooling.layer[k,i,j]
      upsample[k,2*i,2*j-1] <- pooling.layer[k,i,j]
      upsample[k,2*i,2*j] <- pooling.layer[k,i,j]
    }
  }
}
pooling_size <- 2
upsample <- upsample/(pooling_size*pooling_size)
upsample[upsample <= 0] <- 0
dconv <- upsample

dconv_sum <- matrix(0,conv_layer_size,conv_layer_size)
for (i in 1:N) {
  dconv_sum <- dconv[i,,] + dconv_sum
}
dconv_sum <- dconv_sum/N

filter_size <- 3
dW1 <- array(0,c(N,filter_size,filter_size))
temp <- matrix(0,filter_size,filter_size)
for (i in 1:N) {
  dW1[i,,] <- convolution(rotate(rotate(X_train[i,,])), W = dconv[i,,], b = 0)
  temp <- dW1[i,,] + temp
}

dW1 <- temp

db1 <- sum(dconv_sum)

reg <- 1e-1
lr <- 1e-1

# update ....
dW2 <- dW2 + reg*W2
dW1 <- dW1 + reg*W1

W1 <- W1 - lr * dW1
b1 <- b1 - lr * db1

W2 <- W2 - lr * dW2
b2 <- b2 - lr * db2