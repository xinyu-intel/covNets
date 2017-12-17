# rotate matrix
rotate <- function(x) t(apply(x, 2, rev))