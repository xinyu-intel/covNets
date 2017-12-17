# array to col
arr2col <- function(X){
  output <- array(X,c(dim(X)[1],(dim(X)[2]*dim(X)[3]*dim(X)[4])))
  cache <- X
  list(output = output, cache = cache)
}

# col to array
col2arr <- function(X, cache){
  output <- array(X,dim = dim(cache))
}