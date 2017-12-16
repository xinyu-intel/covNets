distribute_value <- function(dz, shape){
  
  # Distributes the input value in the matrix of dimension shape
  # 
  # Arguments:
  # dz -- input scalar
  # shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
  # 
  # Returns:
  # a -- Array of size (n_H, n_W) for which we distributed the value of dz
  
  # Retrieve dimensions from shape
  n_H <- shape[1]
  n_W <- shape[2]
  
  # Compute the value to distribute on the matrix
  average <- dz / (n_H * n_W)
  
  # Create a matrix where every entry is the "average" value
  a = matrix(average, n_H, n_W)
  
  return(a)
}

a = distribute_value(2, c(2,2))