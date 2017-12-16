create_mask_from_window <- function(x){
  
  # Creates a mask from an input matrix x, to identify the max entry of x.
  # 
  # Arguments:
  # x -- Array of shape (f, f)
  # 
  # Returns:
  # mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.

  mask <- (x == max(x)) + 0
}