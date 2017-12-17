# convolution layer
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