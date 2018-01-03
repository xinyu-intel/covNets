library(keras)

# Data Preparation -----------------------------------------------------

batch_size <- 128
num_classes <- 10
epochs <- 12

# Input image dimensions
img_rows <- 28
img_cols <- 28

# The data, shuffled and split between train and test sets

sources <- c("mnist.R")

for (i in 1:length(sources)) {
  cat(paste("Loading ",sources[i],"\n"))
  source(sources[i])
}

provideMNIST(folder = "./data/", download = T)
load("./data/train.RData")
load("./data/test.RData")

dim(trainData) # 60000 784
dim(trainLabels) # 60000 10
dim(testData) # 10000 784
dim(testLabels) # 10000 10

x_train <- trainData
y_train <- trainLabels
x_test <- testData
y_test <- testLabels

# Redefine  dimension of train/test inputs
dim(x_train) <- c(nrow(x_train), 28, 28, 1)
dim(x_test) <- c(nrow(x_test), 28, 28, 1)
input_shape <- c(img_rows, img_cols, 1)

# Transform RGB values into [0,1] range
x_train <- x_train / 255
x_test <- x_test / 255

cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')


# Define Model -----------------------------------------------------------

model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 6, kernel_size = c(5,5), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 16, kernel_size = c(5,5), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 84, activation = 'relu') %>% 
  layer_dense(units = num_classes, activation = 'softmax')

# Compile model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# Train & Evaluate -------------------------------------------------------

model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_data = list(x_test, y_test)
)
scores <- model %>% evaluate(
  x_test, y_test, verbose = 0
)

# Output metrics
cat('Test loss:', scores[[1]], '\n')
cat('Test accuracy:', scores[[2]], '\n')