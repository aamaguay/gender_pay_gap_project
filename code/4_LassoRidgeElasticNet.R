# set wd
setwd("/home/user/Downloads/info_tud/statistical_learning_UDE/gender_pay_gap_project")
setwd("C:/Users/lbergmann/OneDrive - RWI–Leibniz-Institut für Wirtschaftsforschung e.V/Dokumente/Promotion/StatisticalLearning/gender_pay_gap_project_1")

# packages
library(dplyr)
library(GGally)
library(caret)
library(ISLR)
library(tidyverse)
library(caret)
library(glmnet)



#load data 
ds <- read.csv("data/full_data_w_dummies_interaction.csv")
ds <- subset(ds, select = -c(X, X.1) )

#split in train, validation and test data
set.seed(123)

trainindex <- createDataPartition(y = ds$realrinc, p = 0.7, list = FALSE) # Split the data into training (70%) and remaining (30%)
train <- ds[trainindex, ]
remaining <- ds[-trainindex, ]

valid_test_index <- createDataPartition(y = remaining$realrinc, p = 0.5, list = FALSE) # Split the remaining data into validation (50%) and test (50%)
validation <- remaining[valid_test_index, ]
test <- remaining[-valid_test_index, ]


#Lasso
lasso <- train(realrinc ~ female + . - log_realrinc, data = train, 
                method = "glmnet", trControl = trainControl(method = "cv"), 
                tuneGrid = expand.grid(alpha = 1, lambda = seq(0,500,1)))
caret::RMSE(pred = predict(lasso, validation), obs = validation$realrinc) #24283.21


#Ridge
ridge <- train(realrinc ~ female + . - log_realrinc, data = train,
  method = "glmnet", trControl = trainControl(method = "cv"),
  tuneGrid = expand.grid(alpha = 0, lambda = seq(0, 500, 1))) #24325.68
caret::RMSE(pred = predict(ridge, validation), obs = validation$realrinc)

#Elastic Net
elasticnet <- train(realrinc ~ female + . - log_realrinc, data = train,
                    method = "glmnet", trControl = trainControl(method = "cv"),
                    tuneGrid = expand.grid(alpha = seq(from=0, to=1, by = 0.1),
                    lambda = seq(from=0, to=0.15, by = 0.001)))
caret::RMSE(pred = predict(elasticnet, validation), obs = validation$realrinc) #24333.99


write.csv(ds, file = "data/final_data.csv")
