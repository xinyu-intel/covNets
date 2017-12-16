# function padding for 3-D array
pad3d <- function(input=NULL, stride=1) 
{
  input_N <- dim(input)[1]
  input_H <- dim(input)[2]
  input_W <- dim(input)[3]
  input_C <- dim(input)[4]
  output <- array(0,c(input_N,input_H+2*stride,input_W+2*stride,input_C))
  for (i in 1:input_N) {
    for (j in 1:input_C) {
      output[i,,,j] <- rbind(matrix(0,stride,input_W+2*stride),
                           cbind(matrix(0,input_H,stride),input[i,,,j],matrix(0,input_H,stride)),
                           matrix(0,stride,input_W+2*stride))
    }
  }
  return(output)
}