---
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
##          A 1708   48  422   52    2
##          B  394  680  319  125    0
##          C   63   54 1080  158   13
##          D  140   84  676  386    0
##          E   80  334  462    7  559
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5625          
##                  95% CI : (0.5514, 0.5735)
##     No Information Rate : 0.3771          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4458          
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7161  0.56667   0.3650  0.53022  0.97387
## Specificity            0.9040  0.87391   0.9411  0.87356  0.87858
## Pos Pred Value         0.7652  0.44796   0.7895  0.30016  0.38766
## Neg Pred Value         0.8794  0.91783   0.7099  0.94787  0.99766
## Prevalence             0.3040  0.15294   0.3771  0.09279  0.07316
## Detection Rate         0.2177  0.08667   0.1376  0.04920  0.07125
## Detection Prevalence   0.2845  0.19347   0.1744  0.16391  0.18379
## Balanced Accuracy      0.8101  0.72029   0.6530  0.70189  0.92622
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
## 0.8145161 0.6406319 0.6553067 0.7005181 0.5690531
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
##          A 23.2  2.7  1.7  0.9  0.8
##          B  0.7 12.4  1.6  1.1  3.2
##          C  2.3  2.4 11.5  1.9  1.8
##          D  2.1  0.9  2.2 11.4  2.0
##          E  0.1  0.9  0.4  1.0 10.6
##                             
##  Accuracy (average) : 0.6912
```

```r
# Getting confusionMatrix() data in a friendly format
xtab <- confusionMatrix(response, predict(fit.lda))
as.matrix(xtab)
```

```
##      A    B    C    D    E
## A 2740   84  269  246    9
## B  328 1469  285   98   99
## C  201  184 1356  266   47
## D  108  113  229 1360  120
## E   90  376  215  239 1245
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
##          A 2229   13    0    0    0
##          B    2 1504   22    0    0
##          C    0    1 1345   19    0
##          D    1    0    1 1266    5
##          E    0    0    0    1 1437
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9917          
##                  95% CI : (0.9895, 0.9936)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9895          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9987   0.9908   0.9832   0.9844   0.9965
## Specificity            0.9977   0.9962   0.9969   0.9989   0.9998
## Pos Pred Value         0.9942   0.9843   0.9853   0.9945   0.9993
## Neg Pred Value         0.9995   0.9978   0.9965   0.9970   0.9992
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2841   0.1917   0.1714   0.1614   0.1832
## Detection Prevalence   0.2858   0.1947   0.1740   0.1622   0.1833
## Balanced Accuracy      0.9982   0.9935   0.9900   0.9917   0.9982
```

It seems that Randon Forest Model provides the best accuracy level, which is  0.9935.   
Finally, we can look at the most important variable that have higher scores. 


```r
# Calculate the variable importance using the varImp function in the caret package.
varImp(RF) 
```

```
##                        Overall
## pitch_belt           456.96372
## yaw_belt             636.40220
## total_accel_belt     183.31248
## gyros_belt_x          70.35806
## gyros_belt_y          83.79136
## gyros_belt_z         238.70693
## accel_belt_x          83.02730
## accel_belt_y          91.22789
## accel_belt_z         298.63460
## magnet_belt_x        161.52058
## magnet_belt_y        285.06045
## magnet_belt_z        301.21026
## roll_arm             202.57622
## pitch_arm            106.76387
## yaw_arm              146.10777
## total_accel_arm       64.43328
## gyros_arm_x           88.47016
## gyros_arm_y           91.35519
## gyros_arm_z           40.08341
## accel_arm_x          138.35764
## accel_arm_y          100.90650
## accel_arm_z           86.12269
## magnet_arm_x         166.34737
## magnet_arm_y         150.58798
## magnet_arm_z         117.28742
## roll_dumbbell        246.22384
## pitch_dumbbell       115.06552
## yaw_dumbbell         169.28583
## total_accel_dumbbell 164.95140
## gyros_dumbbell_x      86.11187
## gyros_dumbbell_y     169.54116
## gyros_dumbbell_z      58.46280
## accel_dumbbell_x     157.19651
## accel_dumbbell_y     251.75775
## accel_dumbbell_z     205.21409
## magnet_dumbbell_x    311.27009
## magnet_dumbbell_y    418.20516
## magnet_dumbbell_z    471.71122
## roll_forearm         357.82708
## pitch_forearm        471.89681
## yaw_forearm          103.22639
## total_accel_forearm   80.76246
## gyros_forearm_x       50.31034
## gyros_forearm_y       80.85274
## gyros_forearm_z       52.34506
## accel_forearm_x      205.98689
## accel_forearm_y       87.80001
## accel_forearm_z      158.17102
## magnet_forearm_x     132.31484
## magnet_forearm_y     136.38166
## magnet_forearm_z     177.17234
```

```r
# orderin the variable
vi <- RF$importance
order(vi, decreasing=T)
```

```
##  [1]  2 40 38  1 37 39 36 12  9 11 34 26  6 46 35 13  3 51 31 28 23 29 10 48 33
## [26] 24 15 20 50 49 25 27 14 41 21 18  8 17 47 22 30  5  7 44 42  4 16 32 45 43
## [51] 19
```

```r
# What is the order of variable importance?
print(RF$importance)
```

```
##                      MeanDecreaseGini
## pitch_belt                  456.96372
## yaw_belt                    636.40220
## total_accel_belt            183.31248
## gyros_belt_x                 70.35806
## gyros_belt_y                 83.79136
## gyros_belt_z                238.70693
## accel_belt_x                 83.02730
## accel_belt_y                 91.22789
## accel_belt_z                298.63460
## magnet_belt_x               161.52058
## magnet_belt_y               285.06045
## magnet_belt_z               301.21026
## roll_arm                    202.57622
## pitch_arm                   106.76387
## yaw_arm                     146.10777
## total_accel_arm              64.43328
## gyros_arm_x                  88.47016
## gyros_arm_y                  91.35519
## gyros_arm_z                  40.08341
## accel_arm_x                 138.35764
## accel_arm_y                 100.90650
## accel_arm_z                  86.12269
## magnet_arm_x                166.34737
## magnet_arm_y                150.58798
## magnet_arm_z                117.28742
## roll_dumbbell               246.22384
## pitch_dumbbell              115.06552
## yaw_dumbbell                169.28583
## total_accel_dumbbell        164.95140
## gyros_dumbbell_x             86.11187
## gyros_dumbbell_y            169.54116
## gyros_dumbbell_z             58.46280
## accel_dumbbell_x            157.19651
## accel_dumbbell_y            251.75775
## accel_dumbbell_z            205.21409
## magnet_dumbbell_x           311.27009
## magnet_dumbbell_y           418.20516
## magnet_dumbbell_z           471.71122
## roll_forearm                357.82708
## pitch_forearm               471.89681
## yaw_forearm                 103.22639
## total_accel_forearm          80.76246
## gyros_forearm_x              50.31034
## gyros_forearm_y              80.85274
## gyros_forearm_z              52.34506
## accel_forearm_x             205.98689
## accel_forearm_y              87.80001
## accel_forearm_z             158.17102
## magnet_forearm_x            132.31484
## magnet_forearm_y            136.38166
## magnet_forearm_z            177.17234
```

# Conclusion

After checking the Overall Statistics data, the Random Forest model has definitely more accuracy than Decision Tree and LDA models. Hence we will be selecting Random Forest model for final prediction from our data. .












