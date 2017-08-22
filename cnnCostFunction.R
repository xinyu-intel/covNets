cnnCostFunction <- function(probs, model, batchsize){
  W1 <- model$W1
  W2 <- model$W2
  corect.logprobs <- 0
  # compute the loss
  corect.logprobs <- -log(probs[Y.index])
  data.loss <- sum(corect.logprobs)/batchsize
  reg.loss <- 0.5*reg* (sum(W1*W1) + sum(W2*W2))
  loss <- data.loss + reg.loss
}