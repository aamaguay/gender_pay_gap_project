# set wd
setwd("/home/user/Downloads/info_tud/statistical_learning_UDE/gender_pay_gap_project")

# packages
library(dplyr)
library(GGally)

# import ds
ds <- read.csv('data/gss_wages_train.csv')
summary(ds)

cat("the dataset has ", nrow(ds), 'rows, and ', round(sum(is.na(ds$realrinc))/nrow(ds)*100,2), "% of missing values for the outcome \n")

# filtered dataset, not using obs, with missing value for outcome
ds_filter <- ds[!is.na(ds$realrinc),]
summary(ds_filter)

cat("the dataset has ", nrow(ds_filter), 'rows \n')

# fast estimation of correlation matrix
cor.filter <- select_if(ds_filter, is.numeric)
cor.filter <- na.omit(cor.filter)
cor(cor.filter, method = "spearman")

# pair plots
pairs(cor.filter)
ggpairs(cor.filter)
