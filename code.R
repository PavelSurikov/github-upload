
#Download the data
data <- read.csv("C:/Users/Acer_302/Downloads//heart.csv")

#See the provided data
str(data)

#Change name to more understandable
names(data)[1] <- "age"

#In my analysis I well use all 14 of provided variables
#Change the class of target variable to factor 
data$target <- as.factor(data$target)

# Dealing with NAs
colSums(is.na(data))
#There are no NAs in the data

#Is the data imbalanced
table(data$target)
#No, data is well balanced

#Check for outliers:
boxplot(data$age)
boxplot(data$trestbps)
boxplot(data$chol)
boxplot(data$thalach)
boxplot(data$oldpeak)

#There some outliers in these variables: trestbps, chol, thalach, oldpeak
#Handling the outliers
library("DescTools")
data$trestbps <- Winsorize(data$trestbps)
data$chol <- Winsorize(data$chol)
data$thalach <- Winsorize(data$thalach)
data$oldpeak <- Winsorize(data$oldpeak)

#Splitting the data into training and tasting sets
set.seed(111)
index_train = sample(1:nrow(data), 2 / 3 * nrow(data))
training_set = data[index_train, ]
test_set = data[-index_train, ]

#Estimating the logistic model
logmodel <-  glm(target ~ ., family = binomial(link = "logit"), data = training_set)
summary(logmodel)


#Prediction with the cutoff point equals 0.3
test_set$log_pred = predict(logmodel, newdata = test_set, type = 'response')
test_set$log_pred = ifelse(test_set$log_pred < 0.3, 0, 1)

#Evaluating the quality of the model (confusion matrix)
library('caret')
test_set$target = factor(test_set$target)
test_set$log_pred = factor(test_set$log_pred)
mat1 <-  confusionMatrix(test_set$log_pred, test_set$target)
mat1

#Evaluating the quality of the model (AUC)
library(pROC)
ROC_log1 = roc(as.numeric(test_set$target), as.numeric(test_set$log_pred))
plot(ROC_log1)
auc(ROC_log1)

#choice of the cutoff
library(e1071)
library(tidyr)
cutoffs_vec = c()
accur_vec = c()
sens_vec = c()
spec_vec = c()

test_set$log_pred = predict(logmodel, newdata = test_set, type = 'response')
for (i in seq(0, 1, by = 0.05)) {
  print(i)
  cutoffs_vec = c(cutoffs_vec, i)
  test_set$cutoff2 = ifelse(test_set$log_pred < i, 0, 1)
  test_set$cutoff2 = factor(test_set$cutoff2)
  mat1 = confusionMatrix(test_set$cutoff2, test_set$target)
  accur_vec = c(accur_vec, mat1$overall[3])
  sens_vec = c(sens_vec, mat1$byClass[1])
  spec_vec = c(spec_vec, mat1$byClass[2])
  
}

d_loop = data.frame(cutoffs_vec, accur_vec, sens_vec, spec_vec)
d_loop_long = gather(d_loop, key = measure, value = value, -cutoffs_vec)
d_loop_long$measure = factor(d_loop_long$measure, levels=c("accur_vec", "sens_vec", "spec_vec"), 
                             
                             labels=c("Accuracy", "Sensitivity", "Specificity"))

ggplot(data = d_loop_long, aes(x = cutoffs_vec, y = value, col = measure)) +
  geom_line() +
  xlab("Cutoffs") +
  ylab("")

#According to the graph, the best cutoff point is equal 0.625
test_set$log_pred2 = predict(logmodel, newdata = test_set, type = 'response')
test_set$log_pred2 = ifelse(test_set$log_pred2 < 0.625, 0, 1)


#Evaluating the quality of the model with new cutoff point (confusion matrix)
test_set$target = factor(test_set$target)
test_set$log_pred2 = factor(test_set$log_pred2)
mat2 <-  confusionMatrix(test_set$log_pred2, test_set$target)
mat2

#Evaluating the quality of the model with new cutoff point (AUC)
ROC_log2 = roc(as.numeric(test_set$target), as.numeric(test_set$log_pred2))
plot(ROC_log2)
auc(ROC_log2)


#Random forest without cross validation
rf_tt = train(target ~ ., method= "rf", ntree = 100, data = training_set)
test_set$rf_tt <- predict(rf_tt, test_set)

#Evaluating the quality of the model (confusion matrix)
test_set$target = factor(test_set$target)
test_set$rf_tt = factor(test_set$rf_tt)
mat3 <-  confusionMatrix(test_set$rf_tt, test_set$target)
mat3

#Evaluating the quality of the model (AUC)
ROC_rf_tt = roc(test_set$target, as.numeric(test_set$rf_tt))
plot(ROC_rf_tt)
auc(ROC_rf_tt)


#Random forest with cross validation
train.control = trainControl(method = "cv", number = 5)
rf_cv = train(target ~ ., method= "rf", ntree = 50, data = data, trControl = train.control)
data$rf_cv_pred <- predict(rf_cv, data)

#Evaluating the quality of the model (confusion matrix)
data$target = factor(data$target)
data$rf_cv_pred = factor(data$rf_cv_pred)
mat4 <-  confusionMatrix(data$rf_cv_pred, data$target)
mat4

#Evaluating the quality of the model (AUC)
ROC_rf_cv_pred = roc(data$target, as.numeric(data$rf_cv_pred))
plot(ROC_rf_cv_pred)
auc(ROC_rf_cv_pred)




