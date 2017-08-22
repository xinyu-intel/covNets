cnnForward <- function(N, X, model){
  # layer initialization
  C  <- model$C
  P  <- model$P
  K  <- model$K
  W1 <- model$W1
  b1 <- model$b1
  W2 <- model$W2
  b2 <- model$b2
  conv.layer <<- array(0,c(N,C,C))
  pooling.layer <<- array(0,c(N,P,P))
  shaping <<- matrix(0,N,P*P)
  score <<- matrix(0, N, K)
  score.exp <<- matrix(0, N, K)
  probs <<- matrix(0, N, K)
  for (i in 1:N) {
    # forward ....
    # 1 indicate row, 2 indicate col
    conv.layer[i,,] <<- convolution(X[i,,], W = W1, b = b1)
    # neurons : ReLU
    conv.layer[i,,] <<- relu(conv.layer[i,,])
    # pooling layer
    pooling.layer[i,,] <<- meanpool(conv.layer[i,,], 2, 2)
    shaping[i,] <<- matrix(pooling.layer[i,,], nrow = 1)
  }
  # affine layer
  score <<- sweep(shaping %*% W2, 2, b2, '+')
  # softmax
  score.exp <<- exp(score)
  # debug
  probs <<- score.exp/rowSums(score.exp)
  probs
}
