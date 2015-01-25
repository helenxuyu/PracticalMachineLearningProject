Practical Machine Learning Project
========================================================

## Summary of the project

#### The goal of this project is to predict the manner in which they did the exercise by using classification algorithms. The data we used for this project comes from http://groupware.les.inf.puc-rio.br/har.

## Loading the data

```r
# Download the required file
if (!file.exists("/Users/helenxu/Desktop/Coursera/PracticalMachineLearning/pml-training.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","/Users/helenxu/Desktop/Coursera/PracticalMachineLearning/pml-training.csv")}

if (!file.exists("/Users/helenxu/Desktop/Coursera/PracticalMachineLearning/pml-testing.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","/Users/helenxu/Desktop/Coursera/PracticalMachineLearning/pml-testing.csv")}
```


```r
# Load the data into the working directory
training = read.csv("pml-training.csv", na.strings = c("", "NA"))
testing = read.csv("pml-testing.csv", na.strings = c("", "NA"))
```


```r
# Some exploratory data analysis
dim(training)
```

```
## [1] 19622   160
```

```r
dim(testing)
```

```
## [1]  20 160
```

```r
# By looking at the training dataset, we found that there are a lot of NA values.
```

## Processing the data

```r
# We would remove the columns which has NA values in the training/testing dataset.
training = training[,colSums(is.na(training)) == 0]
testing = testing[, colSums(is.na(testing)) == 0]
dim(training);dim(testing)
```

```
## [1] 19622    60
```

```
## [1] 20 60
```

```r
# By looking at the data, we found that the column 1 and 2 is the serial number and the user name, we also want to eliminate them in the new data set.
training = training[, c(3:60)]
testing = testing[,c(3:60)]
```

## Model Fitting
### We would like to use the random forest to do the classification problem.

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.1.2
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
# We would partition the training set into train and test dataset.
inTrain = createDataPartition(y = training$classe, p = 0.7, list = FALSE)
train = training[inTrain,]
test = training[-inTrain,]

# We would use the caret package to fit train the data. We select the method to be random forest and the use the 5-fold cross validation.
modFit = train(classe~., data = train, method = "rf", trControl = trainControl(method = "cv", number = 5, allowParallel = TRUE))
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.1.1
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
print(modFit)
```

```
## Random Forest 
## 
## 13737 samples
##    57 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 10988, 10991, 10989, 10991, 10989 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD   Kappa SD    
##    2    0.9911915  0.9888566  0.0024879748  0.0031477085
##   38    0.9987622  0.9984343  0.0007554001  0.0009555593
##   75    0.9982529  0.9977902  0.0006998511  0.0008852320
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 38.
```

## Out of sample accuracy

```r
testPred = predict(modFit, test)
confusionMatrix(testPred, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    1    0    0    0
##          B    0 1138    0    0    0
##          C    0    0 1026    1    0
##          D    0    0    0  962    1
##          E    0    0    0    1 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9993          
##                  95% CI : (0.9983, 0.9998)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9991          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9991   1.0000   0.9979   0.9991
## Specificity            0.9998   1.0000   0.9998   0.9998   0.9998
## Pos Pred Value         0.9994   1.0000   0.9990   0.9990   0.9991
## Neg Pred Value         1.0000   0.9998   1.0000   0.9996   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1934   0.1743   0.1635   0.1837
## Detection Prevalence   0.2846   0.1934   0.1745   0.1636   0.1839
## Balanced Accuracy      0.9999   0.9996   0.9999   0.9989   0.9994
```

```r
# As we can see form the above statistics, the random forest data classify the data pretty well, the Accuracy is greater than 0.99. We expect the accuracy to be pretty high if the training algorithm is applied to the testing dataset.
```

## Predictions of the testing set

```r
testingPred = predict(modFit, testing)
testingPred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
