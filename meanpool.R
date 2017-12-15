# meanpool layer
# maxpool may be better
meanpool <- function(input_img, ksize, stride, show=FALSE, out=TRUE)
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