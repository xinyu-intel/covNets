softmax <- function(X){

  score.exp <- exp(X)
  probs <- score.exp/rowSums(score.exp)
  
  cache <- X
  list(probs = probs, cache = cache)
  
}