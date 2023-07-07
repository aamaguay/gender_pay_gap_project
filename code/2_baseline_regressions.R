# set wd
setwd("/home/user/Downloads/info_tud/statistical_learning_UDE/gender_pay_gap_project")
setwd("C:/Users/lbergmann/OneDrive - RWI–Leibniz-Institut für Wirtschaftsforschung e.V/Dokumente/Promotion/StatisticalLearning/Project")

# packages
library(dplyr)
library(GGally)
library(caret)

#load data
load("data/ds.Rda")
load("data/ds_na.Rda")
load("data/ds_filter.Rda")

#split in train, validation and test data
set.seed(123)

trainindex_filter <- createDataPartition(y = ds_filter$realrinc, p = 0.7, list = FALSE) # Split the data into training (70%) and remaining (30%)
train_filter <- ds_filter[trainindex_filter, ]
remaining_filter <- ds_filter[-trainindex_filter, ]

valid_test_index_filter <- createDataPartition(y = remaining_filter$realrinc, p = 0.5, list = FALSE) # Split the remaining data into validation (50%) and test (50%)
validation_filter <- remaining_filter[valid_test_index_filter, ]
test_filter <- remaining_filter[-valid_test_index_filter, ]

trainindex_na <- createDataPartition(y = ds_na$realrinc, p = 0.7, list = FALSE) # Split the data into training (70%) and remaining (30%)
train_na <- ds_na[trainindex_na, ]
remaining_na <- ds_na[-trainindex_na, ]

valid_test_index_na <- createDataPartition(y = remaining_na$realrinc, p = 0.5, list = FALSE) # Split the remaining data into validation (50%) and test (50%)
validation_na <- remaining_na[valid_test_index_na, ]
test_na <- remaining_na[-valid_test_index_na, ]

#OLS regressions
m1 <- lm(realrinc ~ . - occrecode -wrkstat - gender -educcat -maritalcat, data = train_na )
summary(m1)

m2 <- with(Data,lm(realrinc ~ . - occrecode -wrkstat - gender -educcat -maritalcat))
summary(pool(m2))
