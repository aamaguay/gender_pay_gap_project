# set wd
tryCatch({
  setwd("C:/Users/lbergmann/OneDrive - RWI–Leibniz-Institut für Wirtschaftsforschung e.V/Dokumente/Promotion/StatisticalLearning/gender_pay_gap_project_1")
}, error = function(err1) {
  # If an error occurs, handle it by setting the working directory to the second path
  setwd("/home/user/Desktop/files_desktop/gender_pay_gap_project")
})
# import functions file
source('code/utils_functions.R')

# packages
library(dplyr)
library(GGally)
library(mice)
library(beepr)
library(collapse)
library(parallel)
library(caret)
library(ggplot2)
library(stringr)
library(ISLR)
library(tidyverse)
library(lightgbm)
library(Matrix)
library(parallel)
library(sgd)
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(data.table)
library(stargazer)
# remotes::install_github("mlr-org/mlr3extralearners@*release")
library(mlr3extralearners)

# import ds
test <- read.csv('data/gss_wages_test.csv')

#check for missings: same problem as with the old dataset
md.pattern(test)

#create factor variables
test$educcat <- as.factor(test$educcat)
test$maritalcat <- as.factor(test$maritalcat)
test$wrkstat <- as.factor(test$wrkstat)
test$gender <- as.factor(test$gender)
test$occrecode <- as.factor(test$occrecode)

# filtered dataset, not using obs, with missing value for outcome
ds_filter  <- test[!is.na(test$realrinc),] #filtered dataset
summary(ds_filter)

ds_filter2 <- ds_filter %>% filter(occ10 != 9997)
cat("the dataset with filter#2 has ", nrow(ds_filter2), 'rows \n')
summary(ds_filter2)

ds_with_filter_col <- ds_filter2 %>%
  select(c("realrinc", "age", "occrecode", "prestg10", "childs","wrkstat", "gender", "educcat", "maritalcat"))

imp.data.filter.col <- mice(ds_with_filter_col, m = 6,
                            method = c("", "pmm", "", "pmm", "pmm", "", "","polyreg","polyreg"),
                            maxit = 5, seed = 1)

# extract the datasets of each iterations
full.datasets <- array(NA, dim = c(nrow(ds_with_filter_col), ncol(ds_with_filter_col), 6))
for (i.ds in 1:6) {
  # Perform imputation for the current imputation number (i.ds)
  imputed_data <- complete(imp.data.filter.col, action = i.ds) %>% as.matrix()
  full.datasets[,,i.ds] <- imputed_data
}
colnames(full.datasets) <- colnames(ds_with_filter_col)

# due to the results of each imputation dataset are not similar (randomness), we need to join these datatsets
# I apply a mean estimation for the numeric values, and majority vote for factor values
ds.after.imp.method <- matrix(NA, ncol = ncol(ds_with_filter_col) , nrow = nrow(ds_with_filter_col))
colnames(ds.after.imp.method) <- colnames(ds_with_filter_col)
for (i.name in colnames(ds_with_filter_col)) {
  # numeric values
  if ( i.name %in% c("age", "prestg10", "childs") ){
    # i.name <- "age"
    ds.mean.feature <- matrix(NA, ncol = 6, nrow = nrow(ds_with_filter_col))
    for(i.idx in 1:6) ds.mean.feature[,i.idx] <- full.datasets[,,i.idx][,i.name]
    ds.mean.feature <- ds.mean.feature %>% as.data.frame() 
    ds.mean.feature <- sapply(ds.mean.feature, function(x) as.numeric(as.character(x))) #here suddenly NAs are included
    ds.mean.feature <- round(rowMeans(ds.mean.feature, na.rm = TRUE))
    ds.after.imp.method[,i.name] <- ds.mean.feature
  }
  # factor values
  else if (i.name %in% c("educcat", "maritalcat")){
    ds.majority.feature <- matrix(NA, ncol = 6, nrow = nrow(ds_with_filter_col))
    for(i.idx in 1:6) ds.majority.feature[,i.idx] <- full.datasets[,,i.idx][,i.name]
    ds.majority.feature <- ds.majority.feature %>% as.data.frame()
    ds.majority.feature.res <- mclapply(as.data.frame(t(ds.majority.feature)), majority_vote, mc.cores = 1) #mc.cores > 1 is not allowed in Windows
    ds.majority.feature.res <- as.vector(unlist(ds.majority.feature.res))
    ds.after.imp.method[,i.name] <- ds.majority.feature.res
  }
}

sum(is.na(ds.after.imp.method))
# fix data type
ds.after.imp.method <- ds.after.imp.method %>% as.data.frame()
ds.after.imp.method <- ds.after.imp.method
ds.after.imp.method$realrinc <- ds_with_filter_col$realrinc
ds.after.imp.method$occrecode <- ds_with_filter_col$occrecode
ds.after.imp.method$wrkstat <- ds_with_filter_col$wrkstat
ds.after.imp.method$gender <- ds_with_filter_col$gender
ds.after.imp.method$age <- as.numeric(as.character(ds.after.imp.method$age))
ds.after.imp.method$childs <- as.numeric(as.character(ds.after.imp.method$childs))
ds.after.imp.method$prestg10 <- as.numeric(as.character(ds.after.imp.method$prestg10))

# comparison between datasets - before and after imputation
summary(ds_filter2)
summary(ds.after.imp.method)

# create dummies for each categorical variable
# occupation
# ds_occ1 <- ds %>% distinct(occ10) #there are 537 different occupations
# ds_occ2 <- ds %>% distinct(occrecode) #there are 12 different occupations: Office, Professional, NA, Business/Finance, Construction, Sales, Transportation, Service, Production, Farming, Installation, Armed Forces

dummies <- dummyVars("~ occrecode + wrkstat + educcat + maritalcat", data = ds.after.imp.method)
ds.with.dummies <- data.frame(predict(dummies, newdata = ds.after.imp.method)) %>% as_tibble()

# fix colnames
fixed.colnames <- gsub("occrecode|wrkstat|educcat|maritalcat", "", colnames(ds.with.dummies) )
fixed.colnames <- gsub("\\.\\.", "_", gsub("\\.", "_", tolower(fixed.colnames) ))
fixed.colnames <- gsub("__", "_",fixed.colnames)
fixed.colnames <- sub("_", "", fixed.colnames)
fixed.colnames[fixed.colnames == "other"] <- "other_wrkstat"
colnames(ds.with.dummies) <- fixed.colnames

# combinate dummies with numeric columns
ds.w.imputed.dummies <- (cbind(ds.after.imp.method[,c("realrinc","age", "prestg10","childs")], 
                               ds.with.dummies)) %>% as_tibble()
ds.w.imputed.dummies$whiteCollar <- ifelse(ds.w.imputed.dummies$office_and_administrative_support == 1 |
                                             ds.w.imputed.dummies$professional == 1 |
                                             ds.w.imputed.dummies$business_finance == 1 |
                                             ds.w.imputed.dummies$sales ==1, 1, 0)
ds.w.imputed.dummies$female <- ifelse(ds.after.imp.method$gender == "Female", 1, 0)
ds.w.imputed.dummies$age_sqr <- (ds.w.imputed.dummies$age)^2
ds.w.imputed.dummies$log_realrinc <- log(ds.w.imputed.dummies$realrinc)
ds.w.imputed.dummies$row_w_na <- as.numeric(apply(ds_with_filter_col, 1, function(row) any(is.na(row))))
ds.w.imputed.dummies$inter_age_prestg <- ds.w.imputed.dummies$age * ds.w.imputed.dummies$prestg10
ds.w.imputed.dummies$inter_age_childs <- ds.w.imputed.dummies$age * ds.w.imputed.dummies$childs
ds.w.imputed.dummies$inter_prestg_childs <- ds.w.imputed.dummies$prestg10 * ds.w.imputed.dummies$childs

cat("the dataset has ", nrow(ds.w.imputed.dummies), 'rows \n')
summary(ds.w.imputed.dummies)

# create quantiles to show interaction plot
ds.w.imputed.dummies$age_qtil <- cut(ds.w.imputed.dummies$age, breaks = c(18, 30, 50, max(ds.w.imputed.dummies$age)),
                                     labels = c("age18until30", "age31until50", "agegreater51"),include.lowest = TRUE)
ds.w.imputed.dummies$childs_qtil <- cut(ds.w.imputed.dummies$childs, breaks = c(0, 2, 5, max(ds.w.imputed.dummies$childs)),
                                        labels = c("childs0until2", "childs3until5", "childsgreater6"),include.lowest = TRUE)
ds.w.imputed.dummies$prestg10_qtil <- cut(ds.w.imputed.dummies$prestg10, breaks = c(16, 30, 50, max(ds.w.imputed.dummies$prestg10)),
                                          labels = c("prestg16until30", "prestg31until50", "prestggreater51"),include.lowest = TRUE)

# create dummies for the previous categorical variables
dummies_intervals <- dummyVars("~ age_qtil + childs_qtil + prestg10_qtil", data = ds.w.imputed.dummies)
ds.w.imputed.dummies <- cbind(ds.w.imputed.dummies,
                              data.frame(predict(dummies_intervals, newdata = ds.w.imputed.dummies)))%>% as_tibble()
colnames(ds.w.imputed.dummies) <- gsub("\\.", "_",colnames(ds.w.imputed.dummies))
colnames(ds.w.imputed.dummies) <- gsub("age_qtil_|childs_qtil_|prestg10_qtil_", "", colnames(ds.w.imputed.dummies) )

# anova test for interactions
model.ts <- lm(realrinc ~ prestg10 + age + (prestg10 * age)+ (prestg10 * childs)+(age * childs), 
               data = ds.w.imputed.dummies)
anova(model.ts)
summary(model.ts)

ggplot(ds.w.imputed.dummies, aes(x = prestg10, y = log_realrinc, color = age_qtil ) ) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE)

# high income:
# age: 51,55,45,48,47
# Business/Finance, Professional, Installation, Maintenance, and Repair, Armed Forces, Construction/Extraction
# prestg: 80,72,75,68,71,70
# childs: 2,3,1,4,0,5
# Full-Time, Temporarily Not Working, Retired
# Graduate , Bachelor, Junior College
# Married, Divorced, Separated
# male, female

# estimate interactions excluding one label of each categorical variable in order to mitigate multicollinearity problems
full.col.names <- colnames(ds.w.imputed.dummies)
occrecode.labels <- full.col.names[match("armed_forces", full.col.names):match("transportation", full.col.names)]
occrecode.labels <- occrecode.labels[occrecode.labels != "armed_forces"]

wrkstat.labels <- full.col.names[match("full_time", full.col.names):match("unemployed_laid_off", full.col.names)]
wrkstat.labels <- wrkstat.labels[wrkstat.labels != "other_wrkstat"]

educcat.labels <- full.col.names[match("bachelor", full.col.names):match("lessthan_high_school", full.col.names)]
educcat.labels <- educcat.labels[educcat.labels != "junior_college"]

maritalcat.labels <- full.col.names[match("divorced", full.col.names):match("widowed", full.col.names)]
maritalcat.labels <- maritalcat.labels[maritalcat.labels != "separated" ]

age.intervals.labels <- full.col.names[match("age18until30", full.col.names):match("agegreater51", full.col.names)]
age.intervals.labels <- age.intervals.labels[age.intervals.labels != "agegreater51" ]

childs.intervals.labels <- full.col.names[match("childs0until2", full.col.names):match("childsgreater6", full.col.names)]
childs.intervals.labels <- childs.intervals.labels[childs.intervals.labels != "childsgreater6" ]

prestg.intervals.labels <- full.col.names[match("prestg16until30", full.col.names):match("prestggreater51", full.col.names)]
prestg.intervals.labels <- prestg.intervals.labels[prestg.intervals.labels != "prestg16until30" ]

# create the combination of features
col_comb_features <- c(paste("occ_", occrecode.labels, sep = ""),
                       paste("wrk_", wrkstat.labels, sep = ""),
                       paste("edu_", educcat.labels, sep = ""),
                       paste("marital_", maritalcat.labels, sep = ""),
                       paste("age_", age.intervals.labels, sep = ""),
                       paste("childs_", childs.intervals.labels, sep = ""),
                       paste("prestg_", prestg.intervals.labels, sep = ""),
                       'female' )
all_comb <- combn(col_comb_features, 2)
cat("total features: ", ncol(all_comb), "\n" )

# remove the combination of the variables from the  the number of combination
ls_ds_interaction <- list()
ls_ds_interaction_pt <- list()
for (ncomb in 1:ncol(all_comb)){
  pt_1 <- all_comb[1,ncomb]
  pt_2 <- all_comb[2,ncomb]
  if ( substr(pt_1, start=1, stop=3) != substr(pt_2, start=1, stop=3) ){
    pt_1 <- gsub("occ_|wrk_|edu_|marital_|age_|childs_|prestg_", "", pt_1 )
    pt_2 <- gsub("occ_|wrk_|edu_|marital_|age_|childs_|prestg_", "", pt_2 )
    ls_ds_interaction[[ncomb]] <- paste(pt_1,pt_2,sep = 'interact')
    ls_ds_interaction_pt[[ncomb]] <- paste(pt_1,pt_2,sep = '.')
  }
}

ls_ds_interaction <- Filter(Negate(is.null), ls_ds_interaction)
ls_ds_interaction <- str_split(ls_ds_interaction,'interact')
cat("filtered total features: ", length(ls_ds_interaction), "\n" )

mx.result.tibble <- do.call(cbind, 
                            lapply(1:length(ls_ds_interaction),
                                   FUN = function(i) estimate.vector(i, ls_ds_interaction, ds.w.imputed.dummies ) ) )
colnames(mx.result.tibble) <- unlist(ls_ds_interaction_pt)
mx.result.tibble <- as_tibble(mx.result.tibble)
# mx.result.tibble <- mx.result.tibble[,(colSums(mx.result.tibble) >0)] 
cat("filtered total features with column sum greater than 0: ", ncol(mx.result.tibble ), "\n" )

full_data_final <- cbind(ds.w.imputed.dummies,
                         ds.after.imp.method %>% select(occrecode,wrkstat,gender,educcat,maritalcat),
                         mx.result.tibble) %>% as_tibble()

# fast estimation of correlation matrix
cor.filter <- select_if(test, is.numeric)
matrix <- cor(cor.filter, method = "spearman")
highCorFeatures <- findCorrelation(matrix, cutoff = 0.9, exact = TRUE)
test <- test[,-highCorFeatures]

# save datasets
write.csv(full_data_final, file = "data/test_final.csv")
rm(list = ls())

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
test <- read.csv("data/test_final.csv")

test$wrkstat <- as.factor(test$wrkstat)
test$educcat <- as.factor(test$educcat)
test$gender <- as.factor(test$gender)
test$occrecode <- as.factor(test$occrecode)
test$maritalcat <- as.factor(test$maritalcat)

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

#OLS
formula <- as.formula(paste("realrinc ~", paste(x_cols_dummys, collapse = " + ")))
set.seed(123)
mod_full <- train(formula,
                  data = train, 
                  method = "lm",  
                  trControl = trainControl(method = "cv", number = 3))
summary(mod_full)
rmse <- caret::RMSE(pred = predict(mod_full, test), obs = test$realrinc)
cat("Root Mean Squared Error (RMSE):", rmse, "\n") #22666.87
cv_results <- mod_full$results$RMSE
avg_rmse <- mean(cv_results)

#log(income) as outcome variable
set.seed(123)
formula2 <- as.formula(paste("log_realrinc ~", paste(x_cols_dummys, collapse = " + ")))
mod_loginc <- train(formula2,
                    data = train, 
                    method = "lm",  
                    trControl = trainControl(method = "cv", number = 3))
summary(mod_loginc)
pred_log_inc <- exp(predict(mod_loginc, newdata = test))
rmse <- sqrt(mean((test$realrinc - pred_log_inc)^2))
cat("Root Mean Squared Error (RMSE):", rmse, "\n") #22874.2
#using log income increases RMSE
cv_results <- mod_full$results$RMSE
avg_rmse <- mean(cv_results)
cat("Root Mean Squared Error (RMSE):", avg_rmse, "\n") #24736,92


#Lasso
formula <- as.formula(paste("log_realrinc ~", paste(x_cols_dummys, collapse = " + ")))
set.seed(123)
lasso <- train(formula, data = train, 
               method = "glmnet", trControl = trainControl(method = "cv", number = 3), 
               tuneGrid = expand.grid(alpha = 1, lambda = seq(0,1,0.001)))
caret::RMSE(pred = exp(predict(lasso, test)), obs = test$realrinc) #22877.65

#Ridge
set.seed(123)
ridge <- train(formula, data = train,
               method = "glmnet", trControl = trainControl(method = "cv", number = 3),
               tuneGrid = expand.grid(alpha = 0, lambda = seq(0,1,0.001)))  #23084.74
caret::RMSE(pred = exp(predict(ridge, test)), obs = test$realrinc)

#Elastic Net
set.seed(123)
elasticnet <- train(formula, data = train,
                    method = "glmnet", trControl = trainControl(method = "cv", number = 3),
                    tuneGrid = expand.grid(alpha = seq(from=0, to=1, by = 0.1),
                                           lambda = seq(from=0, to=0.15, by = 0.001)))
caret::RMSE(pred = exp(predict(elasticnet, test)), obs = test$realrinc) #22877.76
