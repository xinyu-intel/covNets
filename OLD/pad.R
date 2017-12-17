# function padding for 2-D array
pad <- function(input=NULL, stride=1) 
{
  input_N <- dim(input)[1]
  input_H <- dim(input)[2]
  input_W <- dim(input)[3]
  output <- array(0,c(input_N,input_H+2*stride,input_W+2*stride))
  for (i in 1:input_N) {
    output[i,,] <- rbind(matrix(0,stride,input_W+2*stride),
                         cbind(matrix(0,input_H,stride),input[i,,],matrix(0,input_H,stride)),
                         matrix(0,stride,input_W+2*stride))
  }
  return(output)
}