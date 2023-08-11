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
library(beepr)



#load data 
ds <- read.csv("data/full_data_w_dummies_interaction.csv")
ds$wrkstat <- as.factor(ds$wrkstat)
ds$educcat <- as.factor(ds$educcat)
ds$gender <- as.factor(ds$gender)
ds$occrecode <- as.factor(ds$occrecode)
ds$maritalcat <- as.factor(ds$maritalcat)

# split in train, validation and test data
set.seed(123)

trainindex_filter <- createDataPartition(y = ds$realrinc, p = 0.7, list = FALSE) # Split the data into training (70%) and remaining (30%)
train <- ds[trainindex_filter, ]
validation<- ds[-trainindex_filter, ]

# valid_test_index_filter <- createDataPartition(y = remaining_filter$realrinc, p = 0.5, list = FALSE) # Split the remaining data into validation (50%) and test (50%)
# validation_filter <- remaining_filter[valid_test_index_filter, ]
# test_filter <- remaining_filter[-valid_test_index_filter, ]

cat("training dim: ", nrow(train), ", test dim: ", nrow(validation), "\n")

full.col.names <- colnames(ds)
occrecode.labels <- full.col.names[match("armed_forces", full.col.names):match("transportation", full.col.names)]
occrecode.labels <- occrecode.labels[occrecode.labels != "armed_forces"]
educcat.labels <- full.col.names[match("bachelor", full.col.names):match("lessthan_high_school", full.col.names)]
educcat.labels <- educcat.labels[educcat.labels != "junior_college"]
prest.qt.labels <- c("prestg16until30", "prestg31until50", "prestggreater51")
all.label.inter <- c(occrecode.labels, educcat.labels, prest.qt.labels)

x_cols_inter <- list()
for (name.col in full.col.names[c(60:ncol(ds))]){
  split_strings <- strsplit(name.col, ".", fixed = TRUE)
  # Access the individual elements
  first_element <- split_strings[[1]][1]
  second_element <- split_strings[[1]][2]
  if ((first_element %in% all.label.inter) & (second_element %in% all.label.inter)){
    x_cols_inter[[length(x_cols_inter) + 1]] <- paste(first_element, second_element, sep = ".")
  }
}
x_cols_inter <- unlist(x_cols_inter)

# create data sets with lbg format
# select regressors and outcome
# x_cols <- colnames(ds)[match("age", colnames(ds)):match("age_sqr", colnames(ds))]

x_cols <- c(colnames(ds)[match("age", colnames(ds)):match("childs", colnames(ds))],
            "inter_age_prestg","inter_age_childs", "inter_prestg_childs",
            "age_sqr", "wrkstat","educcat","gender", "occrecode", "maritalcat", x_cols_inter)
ytarget <- "realrinc"

x_cols_dummys <- c( colnames(ds)[match("age", colnames(ds)):match("childs", colnames(ds))],
                    colnames(ds)[match("armed_forces", colnames(ds)):match("transportation", colnames(ds))],
                    colnames(ds)[match("full_time", colnames(ds)):match("unemployed_laid_off" , colnames(ds))],
                    colnames(ds)[match("bachelor", colnames(ds)):match("lessthan_high_school" , colnames(ds))],
                    colnames(ds)[match("divorced", colnames(ds)):match("widowed", colnames(ds))],
                    "female", "age_sqr",
                    "inter_age_prestg","inter_age_childs", "inter_prestg_childs", x_cols_inter)
x_cols_dummys <- x_cols_dummys[!x_cols_dummys %in% c("armed_forces", "other_wrkstat", "junior_college", "separated")]



#Lasso
lasso <- train(realrinc ~ female + . - log_realrinc, data = train, 
                method = "glmnet", trControl = trainControl(method = "cv"), 
                tuneGrid = expand.grid(alpha = 1, lambda = seq(0,500,1)))
caret::RMSE(pred = predict(lasso, validation), obs = validation$realrinc) #25287.22


#Ridge
ridge <- train(realrinc ~ female + . - log_realrinc, data = train,
  method = "glmnet", trControl = trainControl(method = "cv"),
  tuneGrid = expand.grid(alpha = 0, lambda = seq(0, 500, 1))) #25320.77
caret::RMSE(pred = predict(ridge, validation), obs = validation$realrinc)

#Elastic Net
elasticnet <- train(realrinc ~ female + . - log_realrinc, data = train,
                    method = "glmnet", trControl = trainControl(method = "cv"),
                    tuneGrid = expand.grid(alpha = seq(from=0, to=1, by = 0.1),
                    lambda = seq(from=0, to=0.15, by = 0.001)))
caret::RMSE(pred = predict(elasticnet, validation), obs = validation$realrinc) #25241.7


write.csv(ds, file = "data/final_data.csv")
