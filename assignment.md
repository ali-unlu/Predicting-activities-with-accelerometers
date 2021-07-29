---
title: "Week 4 Peer review assignment"
author: "Ali Unlu"
date: "7/28/2021"
output: 
  html_document: 
    keep_md: yes
---




# Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

# Data

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:  

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

# Research Questions

One thing that people regularly do is quantify how  much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

**The goal of this project is to predict the manner in which they did the exercise.**         
- This is the "classe" variable in the training set.    
- You may use any of the other variables to predict with.    
- You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did.    
- You will also use your prediction model to predict 20 different test cases.          

 


# Data preparation


```r
# required package
library(tidyverse)
library(caret)
library(mlbench)
library(MASS)
library(randomForest)
```


```r
# data
train <- as_tibble(read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"))
test <- as_tibble(read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"))

#str(train$classe)
# convert outcome variable from character to a factor
train$classe <- as.factor(train$classe) 
```

#### 1. Data cleaning  

Since the data has so many variables (160), we can narrow it down. First of all, we can remve variables, which are having nearly zero variance. nearZeroVar() is a function to remove predictors that are sparse and highly unbalanced.   


```r
Zero <- nearZeroVar(train)

train1 <- train[,-Zero]
test1 <- test[,-Zero]

 dim(train1)
```

```
## [1] 19622   100
```

We have now 100 variables but when we checked the data, there are so much missing values in some variables. Another approach might be removing variables which has more than %5 NAs. 


```r
# Removing Variables which are having NA values. Our threshhold is 95%.
NAs <- sapply(train1, function(x) mean(is.na(x))) > 0.95

train2 <- train1[,NAs == FALSE]
test2 <- test1[,NAs == FALSE]
dim(train2);dim(test2)
```

```
## [1] 19622    59
```

```
## [1] 20 59
```

We have now 59 variables but when checking closely the rest of the variables, the first seven of them do not seem to be relevant to our analysis. We finalize data preparation by removing those variables from the set and we will start analyzing the rest of the 52 variables.  


```r
# remove the first 7 variables
clean_train_data <- train2[,-c(1:7)]
clean_test_data <- test2[,-c(1:7)]

# dim(clean_train_data);dim(clean_test_data)
```

#### 2.  Data Partitioning 

For the analysis, we need to split the data into two parts. There are different approaches in the field but we follow the course recommendation (%60 vs %40). SO, the training set will consist 60% of the total data while the test set will have 40% of the total data. 


```r
inTrainIndex <- createDataPartition(clean_train_data$classe, p=0.60)[[1]]
ttrain <- clean_train_data[inTrainIndex,]
ctrain<- clean_train_data[-inTrainIndex,]
dim(ctrain);dim(ttrain)
```

```
## [1] 7846   52
```

```
## [1] 11776    52
```

As seen above, there are 7846 observations in the test set and 11776 in the training set. 

# Model testing 

To respond the project question, we can use different models to test the accuracy of the model they provide. 

### 1. Decision Tree Model


```r
TREE <- train(classe ~ ., method = "rpart", data = ttrain)

TREE_pre <- predict(TREE, ctrain)
confusionMatrix(ctrain$classe,TREE_pre)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2038   34  112   27   21
##          B  668  498  294   58    0
##          C  645   40  528  155    0
##          D  583  232  147  324    0
##          E  326  285  334   76  421
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4855          
##                  95% CI : (0.4744, 0.4966)
##     No Information Rate : 0.543           
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.3261          
##                                           
##  Mcnemar's Test P-Value : <2e-16          
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.4784  0.45730   0.3731  0.50625  0.95249
## Specificity            0.9459  0.84905   0.8694  0.86650  0.86210
## Pos Pred Value         0.9131  0.32806   0.3860  0.25194  0.29196
## Neg Pred Value         0.6042  0.90661   0.8631  0.95183  0.99672
## Prevalence             0.5430  0.13880   0.1803  0.08157  0.05633
## Detection Rate         0.2598  0.06347   0.0673  0.04129  0.05366
## Detection Prevalence   0.2845  0.19347   0.1744  0.16391  0.18379
## Balanced Accuracy      0.7122  0.65317   0.6213  0.68638  0.90730
```

The accuracy rate of the Decision Tree Model is 0.6323, which is lower then our expectations. 

### 2. Linear Discriminant Analysis ("lda") Model 


```r
# applying ld function from the MASS package
predictors <- ttrain[1:51]
response <- ttrain$classe

ld <- lda(predictors, response, CV=TRUE)
ct <- table(response, ld$class)
diag(prop.table(ct,1))
```

```
##         A         B         C         D         E 
## 0.8163082 0.6353664 0.6577410 0.7056995 0.5806005
```

```r
# Applying caret::confusionMatrix()
control <- trainControl(method="LGOCV",number=20)
metric<-"Accuracy"
set.seed(2000)
fit.lda <- train(predictors,response,method="lda",metric=metric,trControl=control)
confusionMatrix(fit.lda)
```

```
## Repeated Train/Test Splits Estimated (20 reps, 75%) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 23.1  2.9  1.8  1.0  0.7
##          B  0.8 12.3  1.6  1.1  3.1
##          C  2.2  2.5 11.5  1.8  1.8
##          D  2.3  0.8  2.1 11.5  1.9
##          E  0.1  0.8  0.4  1.0 10.8
##                             
##  Accuracy (average) : 0.6927
```

```r
# Getting confusionMatrix() data in a friendly format
xtab <- confusionMatrix(response, predict(fit.lda))
as.matrix(xtab)
```

```
##      A    B    C    D    E
## A 2741   88  241  267   11
## B  337 1459  294   89  100
## C  214  187 1359  250   44
## D  110  113  224 1367  116
## E   90  357  219  232 1267
```

### 3. Random Forest Model


```r
RF <- randomForest(classe~., data=ttrain, type="response")

# Perform prediction
RF_pre <- predict(RF, newdata= ctrain, type = "class")

# Following confusion matrix shows the errors of the prediction algorithm.
confusionMatrix(RF_pre, ctrain$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2228   12    0    0    0
##          B    3 1503   10    0    0
##          C    0    3 1352   18    2
##          D    0    0    6 1268    3
##          E    1    0    0    0 1437
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9926          
##                  95% CI : (0.9905, 0.9944)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9906          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9901   0.9883   0.9860   0.9965
## Specificity            0.9979   0.9979   0.9964   0.9986   0.9998
## Pos Pred Value         0.9946   0.9914   0.9833   0.9930   0.9993
## Neg Pred Value         0.9993   0.9976   0.9975   0.9973   0.9992
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2840   0.1916   0.1723   0.1616   0.1832
## Detection Prevalence   0.2855   0.1932   0.1752   0.1628   0.1833
## Balanced Accuracy      0.9980   0.9940   0.9924   0.9923   0.9982
```

It seems that Randon Forest Model provides the best accuracy level, which is  0.9935.   
Finally, we can look at the most important variable that have higher scores. 


```r
# Calculate the variable importance using the varImp function in the caret package.
varImp(RF) 
```

```
##                        Overall
## pitch_belt           454.30576
## yaw_belt             621.53272
## total_accel_belt     162.66808
## gyros_belt_x          69.03275
## gyros_belt_y          81.66229
## gyros_belt_z         252.65476
## accel_belt_x          79.80673
## accel_belt_y          97.40588
## accel_belt_z         310.71109
## magnet_belt_x        172.32078
## magnet_belt_y        294.50868
## magnet_belt_z        263.45677
## roll_arm             191.01957
## pitch_arm            110.65702
## yaw_arm              151.82747
## total_accel_arm       66.48443
## gyros_arm_x           84.75751
## gyros_arm_y           91.20734
## gyros_arm_z           41.94285
## accel_arm_x          154.06082
## accel_arm_y          106.41943
## accel_arm_z           90.14518
## magnet_arm_x         152.39999
## magnet_arm_y         151.98838
## magnet_arm_z         125.18032
## roll_dumbbell        259.99740
## pitch_dumbbell       118.85687
## yaw_dumbbell         167.29442
## total_accel_dumbbell 171.88520
## gyros_dumbbell_x      87.26448
## gyros_dumbbell_y     159.01356
## gyros_dumbbell_z      53.60439
## accel_dumbbell_x     151.76618
## accel_dumbbell_y     268.30690
## accel_dumbbell_z     213.20118
## magnet_dumbbell_x    306.74306
## magnet_dumbbell_y    410.00011
## magnet_dumbbell_z    486.19803
## roll_forearm         360.60551
## pitch_forearm        469.14258
## yaw_forearm          102.87463
## total_accel_forearm   71.06309
## gyros_forearm_x       51.46594
## gyros_forearm_y       81.89341
## gyros_forearm_z       54.97879
## accel_forearm_x      188.99495
## accel_forearm_y       92.75037
## accel_forearm_z      152.09483
## magnet_forearm_x     140.49018
## magnet_forearm_y     137.19401
## magnet_forearm_z     173.88956
```

```r
# orderin the variable
vi <- RF$importance
order(vi, decreasing=T)
```

```
##  [1]  2 38 40  1 37 39  9 36 11 34 12 26  6 35 13 46 51 10 29 28  3 31 20 23 48
## [26] 24 15 33 49 50 25 27 14 21 41  8 47 18 22 30 17 44  5  7 42  4 16 45 32 43
## [51] 19
```

```r
# What is the order of variable importance?
print(RF$importance)
```

```
##                      MeanDecreaseGini
## pitch_belt                  454.30576
## yaw_belt                    621.53272
## total_accel_belt            162.66808
## gyros_belt_x                 69.03275
## gyros_belt_y                 81.66229
## gyros_belt_z                252.65476
## accel_belt_x                 79.80673
## accel_belt_y                 97.40588
## accel_belt_z                310.71109
## magnet_belt_x               172.32078
## magnet_belt_y               294.50868
## magnet_belt_z               263.45677
## roll_arm                    191.01957
## pitch_arm                   110.65702
## yaw_arm                     151.82747
## total_accel_arm              66.48443
## gyros_arm_x                  84.75751
## gyros_arm_y                  91.20734
## gyros_arm_z                  41.94285
## accel_arm_x                 154.06082
## accel_arm_y                 106.41943
## accel_arm_z                  90.14518
## magnet_arm_x                152.39999
## magnet_arm_y                151.98838
## magnet_arm_z                125.18032
## roll_dumbbell               259.99740
## pitch_dumbbell              118.85687
## yaw_dumbbell                167.29442
## total_accel_dumbbell        171.88520
## gyros_dumbbell_x             87.26448
## gyros_dumbbell_y            159.01356
## gyros_dumbbell_z             53.60439
## accel_dumbbell_x            151.76618
## accel_dumbbell_y            268.30690
## accel_dumbbell_z            213.20118
## magnet_dumbbell_x           306.74306
## magnet_dumbbell_y           410.00011
## magnet_dumbbell_z           486.19803
## roll_forearm                360.60551
## pitch_forearm               469.14258
## yaw_forearm                 102.87463
## total_accel_forearm          71.06309
## gyros_forearm_x              51.46594
## gyros_forearm_y              81.89341
## gyros_forearm_z              54.97879
## accel_forearm_x             188.99495
## accel_forearm_y              92.75037
## accel_forearm_z             152.09483
## magnet_forearm_x            140.49018
## magnet_forearm_y            137.19401
## magnet_forearm_z            173.88956
```

# Conclusion

After checking the Overall Statistics data, the Random Forest model has definitely more accuracy than Decision Tree and LDA models. Hence we will be selecting Random Forest model for final prediction from our data. .

# Course Project Prediction Quiz Portion

Apply your machine learning algorithm to the 20 test cases available in the test data above and submit your predictions in appropriate format to the Course Project Prediction Quiz for automated grading. 


```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```











