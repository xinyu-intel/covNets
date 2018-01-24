logsumexp <- function (x) {
  y <- max(x)
  y <- y + log(rowSums(exp(x - y)) + 1e-300)
}

softmax <- function (x) {
  probs <- exp(x - logsumexp(x))
  cache <- x
  list(probs = probs, cache = cache)
}