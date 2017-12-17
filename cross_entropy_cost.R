cross_entropy_cost <- function(AL, Y, batch_size){
  
  # create index for both row and col
  Y.set   <- sort(unique(Y))
  Y.index <- cbind(1:batch_size, match(Y, Y.set))
  
  corect.logprobs <- -log(AL[Y.index])
  loss  <- sum(corect.logprobs)/batch_size
  
  cache <- list(AL = AL, Y.index = Y.index)
  list(loss = loss, cache = cache)
  
}