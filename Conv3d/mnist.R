readMNIST <- function(folder)
{
  futile.logger::flog.info("Loading the MNIST data set.")
  
  # This function reads the data and labels from the two files given by
  # dataName and labelName. Afterwards it puts the data and labels
  # together in one matrix and sorts it by the labels. The label is in
  # the last column. Then it returns the sorted matrix.
  loadData <- function(dataName, labelName)
  {
    fileFunction <- file
    
    # Switch to gzfile function if necessary
    if (file.exists(paste0(dataName,".gz")))
    {
      dataName <- paste0(dataName, ".gz")
      labelName <- paste0(labelName, ".gz")
      fileFunction <- gzfile
    }
    
    # Read the data
    file <- fileFunction(dataName,'rb')
    readBin(file,'integer',n=1,size=4,endian='big')
    rows <- readBin(file,'integer',n=1,size=4,endian='big')
    numRow <- readBin(file,'integer',n=1,size=4,endian='big')
    numCol <- readBin(file,'integer',n=1,size=4,endian='big')
    columns <- numRow*numCol
    buffer <- readBin(file,'integer',n=rows*columns,size=1,signed=F)
    data <- matrix(buffer, nrow=rows, byrow=T)/255
    rm(buffer)
    close(file)
    gc()
    
    # Read the labels
    file <- fileFunction(labelName,'rb')
    readBin(file,'integer',n=1,size=4,endian='big')
    num <- readBin(file,'integer',n=1,size=4,endian='big')
    labels <- readBin(file,'integer',n=num,size=1,signed=F)+1
    close(file)
    gc()
    
    # Sort the data by the labels
    sortedData <- cbind(data[],labels[]) # putting data and labels together
    sortedData <- sortedData[order(sortedData[,columns+1]),] # sort the data by the column 785 (the label)
    
    rm(data)
    rm(labels)
    gc()
    return(sortedData)
  }
  
  # Bring the sorted data matrices in a random order
  generateData <- function(data,random,dims){
    # Mix the train data
    randomData <- cbind(data[],random)
    randomData <- randomData[order(randomData[,dims[2]+1]),]
    rdata <- randomData[,1:(dims[2]-1)]
    return(rdata)
  }
  
  generateLabels <- function(counts,random,rows){
    # generate a label matrix with rows of kind c(1,0,0,0,0,0,0,0,0,0) and 
    # mix the train labels
    rlabels <- matrix(0, nrow=rows, ncol=10)
    start <- 1
    end <- 0
    for(i in 1:10){
      c <- rep(0,10) 
      c[i] <- 1
      l <- matrix(c,nrow=counts[i],ncol=10,byrow=TRUE)
      end <- end + counts[i]
      rlabels[start:end,] <- l
      start <- start + counts[i]
      futile.logger::flog.info(paste0("class ", (i-1)," = ", counts[i], " images"))
    } 
    
    randomLabels <- cbind(rlabels, random)
    randomLabels <- randomLabels[order(randomLabels[,11]),]
    rlabels <- randomLabels[,1:10]
    return(rlabels)
  }
  
  futile.logger::flog.info("Loading train set with 60000 images.")
  train <- loadData(paste(folder,"train-images-idx3-ubyte",sep=""), paste(folder,"train-labels-idx1-ubyte",sep=""))
  dims <- dim(train)
  random <- sample(1:dims[1])
  counts <- table(train[,dims[2]])
  futile.logger::flog.info("Generating randomized data set and label matrix")
  trainData <- generateData(train,random,dims)		
  trainLabels <- generateLabels(counts,random,dims[1])
  futile.logger::flog.info("Saving the train data (filename=train)")
  save(trainData, trainLabels, file=paste0(folder, "train.RData"), precheck=T, compress=T)
  
  futile.logger::flog.info("Loading test set with 10000 images.")
  test <- loadData(paste(folder,"t10k-images-idx3-ubyte",sep=""),paste(folder,"t10k-labels-idx1-ubyte",sep=""))
  dims <- dim(test)
  random <- sample(1:dims[1])
  counts <- table(test[,dims[2]])
  futile.logger::flog.info("Generating randomized data set and label matrix")
  testData <- generateData(test,random,dims)		
  testLabels <- generateLabels(counts,random,dims[1])
  print(paste("Saving the test data (filename=test)"))
  save(testData, testLabels, file=paste0(folder, "test.RData"), precheck=T, compress=T)
  futile.logger::flog.info("Finished")
}

provideMNIST <- function(folder="data/", download = F)
{
  # TODO: does not work on windows, will generate warning message because it
  # tries to create the directory even if it exists
  # TODO: make trailing slash optional
  if (!file.exists(folder))
  {
    dir.create(folder)
  }
  
  fileNameTrainImages <- "train-images-idx3-ubyte.gz"
  fileNameTrainLabels <- "train-labels-idx1-ubyte.gz"
  fileNameTestImages <- "t10k-images-idx3-ubyte.gz"
  fileNameTestLabels <- "t10k-labels-idx1-ubyte.gz"
  
  mnistUrl <- "http://yann.lecun.com/exdb/mnist/"
  
  if (file.exists(paste0(folder, "train.RData")) &&
      file.exists(paste0(folder, "test.RData")))
  {
    futile.logger::flog.info("MNIST data set already available, nothing left to do.")
    return(T)
  }
  
  if (download && any(
    !file.exists(paste0(folder,fileNameTrainImages)),
    !file.exists(paste0(folder,fileNameTrainLabels)),
    !file.exists(paste0(folder,fileNameTestImages)),
    !file.exists(paste0(folder,fileNameTestLabels))
  ))
  {
    futile.logger::flog.info("Compressed MNIST files not found, attempting to download...")
    
    statusCodes <- c()
    
    for (file in c(fileNameTrainImages, fileNameTrainLabels,
                   fileNameTestImages, fileNameTestLabels))
    {
      statusCodes <- c(statusCodes,
                       utils::download.file(paste0(mnistUrl, file),
                                            paste0(folder, file)))
    }
    
    if (any(statusCodes > 0))
    {
      futile.logger::flog.error(paste("Error downloading MNIST files.",
                                      "Download manually from %s or try again."), mnistUrl) 
      return(F)
    }
    
    futile.logger::flog.info("Successfully downloaded compressed MNIST files.")
  }
  else
  {
    futile.logger::flog.info(
      "Compressed MNIST files found or download disabled, skipping download.")
  }
  
  readMNIST(folder)
}