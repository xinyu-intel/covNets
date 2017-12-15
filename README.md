# Description

The repo contains some r scripts aiming to build a demo covNets framework for image recognizing from scratch. 

## File Description

**cnn.R** run model on mnist dataset.

**convolution.R** convolution function.

**meanpool.R** mean pooling function.

**mnist.R** download and read mnist dataset automatically(download into data folder and save as .RData file).

**model.R** cnn model, forward and backward.

**predict.R** cnn predict function.

**relu.R** relu activation function.

**rotate.R** rotate matrix.

**python** a cnn python script based mainly on numpy.

**data** mnist data created by mnist.R.

## Problems

- backward function can be wrong.

- With python I can store and append parameters and some cache in a dictionary. How to achieve this goal in R?(list function in R will cause nest problem, e.g. list1(list2(),list3(list4()..)))