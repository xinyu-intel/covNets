upsample <- function(N, conv_layer_size, pooling_layer_size, pooling_size){
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
  upsample <- upsample/(pooling_size*pooling_size)
  upsample
}