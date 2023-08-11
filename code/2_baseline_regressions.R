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


#OLS regressions
mod_full <- train(realrinc ~ female + . - log_realrinc - transportation - unemployed_laid_off - lessthan_high_school - widowed - whiteCollar - age_sqr,
                  data = train, 
                  method = "lm",  
                  trControl = trainControl(method = "cv"))
summary(mod_full)
rmse <- caret::RMSE(pred = predict(mod_full, validation), obs = validation$realrinc)
cat("Root Mean Squared Error (RMSE):", rmse, "\n") #24429.79
###I receive warnings due to multicollinearity

mod_wo_interactions <-train(realrinc ~ female + age + prestg10 + childs + armed_forces + business_finance + construction_extraction + farming_fishing_and_forestry + 
                            installation_maintenance_and_repair + office_and_administrative_support + production + professional + sales + service + 
                             + full_time + housekeeper + part_time + retired + school + temporarily_not_working + unemployed_laid_off + bachelor + 
                            graduate + highschool + juniorcollege +  + divorced + married + nevermarried + separated,
                            data = train, 
                            method = "lm",  
                            trControl = trainControl(method = "cv"))
summary(mod_wo_interactions)
rmse <- caret::RMSE(pred = predict(mod_wo_interactions, validation), obs = validation$realrinc)
cat("Root Mean Squared Error (RMSE):", rmse, "\n") #24611.53


exclude_vars <- grepl( "age_sqr|log_realrinc|age_qtil|childs_qtil|prestg10_qtil|age18until30|age31until50|agegreater51|childs0until2|childs3until5|childsgreater6|prestg16until30|prestg31until50|prestggreater51" , names(train))
train_wo_cat <- train[, !exclude_vars]
mod_wo_cat <- mod_full <- train(realrinc ~ female + . - transportation - unemployed_laid_off - lessthan_high_school - widowed - whiteCollar,
                                data = train_wo_cat, 
                                method = "lm",  
                                trControl = trainControl(method = "cv"))
summary(mod_wo_cat)
rmse <- caret::RMSE(pred = predict(mod_wo_cat, validation), obs = validation$realrinc)
cat("Root Mean Squared Error (RMSE):", rmse, "\n") #24444.66


mod_loginc <- train(log_realrinc ~ female + age + prestg10 + childs + armed_forces + business_finance + construction_extraction + farming_fishing_and_forestry + 
                      installation_maintenance_and_repair + office_and_administrative_support + production + professional + sales + service + 
                      + full_time + housekeeper + part_time + retired + school + temporarily_not_working + unemployed_laid_off + bachelor + 
                      graduate + highschool + juniorcollege +  + divorced + married + nevermarried + separated,
                    data = train, 
                    method = "lm",  
                    trControl = trainControl(method = "cv"))
summary(mod_loginc)
pred_log_inc <- exp(predict(mod_loginc, newdata = validation))
rmse <- sqrt(mean((validation$realrinc - pred_log_inc)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n") #25126.29
#using log income increases RMSE


mod_age2 <- train(realrinc ~ female + age + age_sqr +  prestg10 + childs + armed_forces + business_finance + construction_extraction + farming_fishing_and_forestry + 
                    installation_maintenance_and_repair + office_and_administrative_support + production + professional + sales + service + 
                    + full_time + housekeeper + part_time + retired + school + temporarily_not_working + unemployed_laid_off + bachelor + 
                    graduate + highschool + juniorcollege +  + divorced + married + nevermarried + separated, 
                  data = train, 
                  method = "lm",  
                  trControl = trainControl(method = "cv"))
summary(mod_age2)
rmse <- caret::RMSE(pred = predict(mod_age2, validation), obs = validation$realrinc)
cat("Root Mean Squared Error (RMSE):", rmse, "\n") #24535.69
#reduces RMSE --> include age_sqr

mod_agecat <- train(realrinc ~ female + age18until30 + age31until50 + agegreater51 + prestg10 + childs + armed_forces + business_finance + construction_extraction + farming_fishing_and_forestry + 
                      installation_maintenance_and_repair + office_and_administrative_support + production + professional + sales + service + 
                      + full_time + housekeeper + part_time + retired + school + temporarily_not_working + unemployed_laid_off + bachelor + 
                      graduate + highschool + juniorcollege +  + divorced + married + nevermarried + separated, 
                    data = train, 
                    method = "lm",  
                    trControl = trainControl(method = "cv"))
summary(mod_agecat)
rmse <- caret::RMSE(pred = predict(mod_agecat, validation), obs = validation$realrinc)
cat("Root Mean Squared Error (RMSE):", rmse, "\n") #24582.74 
#higher RMSE than for age_sqr

mod_ageqtil <- train(realrinc ~ female + age_qtil+ prestg10 + childs + armed_forces + business_finance + construction_extraction + farming_fishing_and_forestry + 
                    installation_maintenance_and_repair + office_and_administrative_support + production + professional + sales + service + 
                    + full_time + housekeeper + part_time + retired + school + temporarily_not_working + unemployed_laid_off + bachelor + 
                    graduate + highschool + juniorcollege +  + divorced + married + nevermarried + separated, 
                  data = train, 
                  method = "lm",  
                  trControl = trainControl(method = "cv"))
summary(mod_ageqtil)
rmse <- caret::RMSE(pred = predict(mod_ageqtil, validation), obs = validation$realrinc)
cat("Root Mean Squared Error (RMSE):", rmse, "\n") #24582.74 
#higher RMSE than for age_sqr

mod_whitecollar <- mod_age2 <- train(realrinc ~ female + age +  prestg10 + childs + armed_forces + whiteCollar + 
                                       + full_time + housekeeper + part_time + retired + school + temporarily_not_working + unemployed_laid_off + bachelor + 
                                       graduate + highschool + juniorcollege +  + divorced + married + nevermarried + separated, 
                                     data = train, 
                                     method = "lm",  
                                     trControl = trainControl(method = "cv"))
summary(mod_whitecollar)
rmse <- caret::RMSE(pred = predict(mod_whitecollar, validation), obs = validation$realrinc)
cat("Root Mean Squared Error (RMSE):", rmse, "\n") #24817.7
#do include occupations separately


###final dataset
exclude_vars <- grepl( "log_realrinc|age_qtil|age18until30|age31until50|agegreater51|childs0until2|childs3until5|childsgreater6|prestg16until30|prestg31until50|prestggreater51|whiteCollar" , names(ds))
ds_final <- ds[, !exclude_vars]

set.seed(123)

trainindex <- createDataPartition(y = ds_final$realrinc, p = 0.7, list = FALSE) # Split the data into training (70%) and remaining (30%)
train <- ds_final[trainindex, ]
remaining <- ds_final[-trainindex, ]

valid_test_index <- createDataPartition(y = remaining$realrinc, p = 0.5, list = FALSE) # Split the remaining data into validation (50%) and test (50%)
validation <- remaining[valid_test_index, ]
test <- remaining[-valid_test_index, ]

mod_baseline <- train(realrinc ~ female + age + age_sqr +  prestg10 + childs + armed_forces + business_finance + construction_extraction + farming_fishing_and_forestry + 
                                    installation_maintenance_and_repair + office_and_administrative_support + production + professional + sales + service + 
                                    + full_time + housekeeper + part_time + retired + school + temporarily_not_working + unemployed_laid_off + bachelor + 
                                    graduate + highschool + juniorcollege +  + divorced + married + nevermarried + separated, 
                                  data = train, 
                                  method = "lm",  
                                  trControl = trainControl(method = "cv"))
summary(mod_baseline)
rmse <- caret::RMSE(pred = predict(mod_baseline, validation), obs = validation$realrinc)
cat("Root Mean Squared Error (RMSE):", rmse, "\n") #24535.69
