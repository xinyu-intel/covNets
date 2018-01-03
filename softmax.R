logsumexp <- function (x) {
  y <- max(x)
  y <- y + log(sum(exp(x - y)))
}

softmax <- function (x) {
  probs <- exp(x - logsumexp(x))
  cache <- x
  list(probs = probs, cache = cache)
}