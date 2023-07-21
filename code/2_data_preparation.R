# set wd
setwd("/home/user/Downloads/info_tud/statistical_learning_UDE/gender_pay_gap_project")
# import functions file
source('code/utils_functions.R')
# setwd("C:/Users/lbergmann/OneDrive - RWI–Leibniz-Institut für Wirtschaftsforschung e.V/Dokumente/Promotion/StatisticalLearning/Project")

# packages
library(dplyr)
library(GGally)
library(mice)
library(beepr)
library(collapse)
library(parallel)
library(caret)
library(ggplot2)

# import ds
ds <- read.csv('data/gss_wages_train.csv')
summary(ds)
ds1 <- ds
cat("the dataset has ", nrow(ds), 'rows, and ', round(sum(is.na(ds$realrinc))/nrow(ds)*100,2), "% of missing values for the outcome \n")

# filtered dataset, not using obs, with missing value for outcome
ds_filter  <- ds[!is.na(ds$realrinc),] #filtered dataset
summary(ds_filter)

cat("the dataset has ", nrow(ds_filter), 'rows \n')
summary(ds_filter)

# NA analysis for filtered dataset: 
# 33066 obs for which we have all variables, 
# 382 obs miss prestige
# 82 miss occupation and prestige
# 71 miss children
md.pattern(ds_filter)

# NA analysis: children
ds_child <- ds_filter %>% filter(is.na(childs))
summary(ds_child)
# those for whom we don't know whether they have children are mainly in their 30s
# male, studied till the high school, Never Married, and work full-time

# NA analysis: prestige
ds_prest <- ds_filter %>% filter(is.na(prestg10))
summary(ds_prest)
# most people for whom we don't know the prestige have code 9997 which implies that 
# their occupation cannot be coded --> I would suggest to remove them

ds_filter2 <- ds_filter %>% filter(occ10 != 9997)
cat("the dataset with filter#2 has ", nrow(ds_filter2), 'rows \n')
summary(ds_filter2)

ds_with_filter_col <- ds_filter2 %>%
  select(c("realrinc", "age", "occrecode", "prestg10", "childs","wrkstat", "gender", "educcat", "maritalcat"))

imp.data.filter.col <- mice(ds_with_filter_col, m = 6,
                            method = c("", "pmm", "", "pmm", "pmm", "", "","polyreg","polyreg"),
                            maxit = 5, seed = 1)
# check performance
# look kinda similar
densityplot(imp.data.filter.col)
# points are located in the same range values
stripplot(imp.data.filter.col, pch = 20, cex = 1.2)
# i dont know why i cant see the red dots
xyplot(imp.data.filter.col, realrinc ~ age+childs+prestg10, pch = c(1, 1), 
       cex = c(1, 1)) 

# extract the datasets of each iterations
full.datasets <- array(NA, dim = c(nrow(ds_with_filter_col), ncol(ds_with_filter_col), 6))
for (i.ds in 1:6) {
  # Perform imputation for the current imputation number (i.ds)
  imputed_data <- complete(imp.data.filter.col, action = i.ds) %>% as.matrix()
  full.datasets[,,i.ds] <- imputed_data
}
colnames(full.datasets) <- colnames(ds_with_filter_col)

# due to the results of each imputation dataset are not similar (randomness), we need to join this datatset
# i apply a mean estimation for the numeric values, and majority vote for factor values
ds.after.imp.method <- matrix(NA, ncol = ncol(ds_with_filter_col) , nrow = nrow(ds_with_filter_col))
colnames(ds.after.imp.method) <- colnames(ds_with_filter_col)
for (i.name in colnames(ds_with_filter_col)) {
  # numeric values
  if ( i.name %in% c("age", "prestg10", "childs") ){
    # i.name <- "childs"
    ds.mean.feature <- matrix(NA, ncol = 6, nrow = nrow(ds_with_filter_col))
    for(i.idx in 1:6) ds.mean.feature[,i.idx] <- full.datasets[,,i.idx][,i.name]
    ds.mean.feature <- ds.mean.feature %>% as.data.frame()
    ds.mean.feature <- (as.data.frame(lapply(ds.mean.feature, function(x) as.numeric(levels(x))[x])))
    ds.mean.feature <- round(rowMeans(ds.mean.feature))
    ds.after.imp.method[,i.name] <- ds.mean.feature
  }
  # factor values
  else if (i.name %in% c("educcat", "maritalcat")){
    ds.majority.feature <- matrix(NA, ncol = 6, nrow = nrow(ds_with_filter_col))
    for(i.idx in 1:6) ds.majority.feature[,i.idx] <- full.datasets[,,i.idx][,i.name]
    ds.majority.feature <- ds.majority.feature %>% as.data.frame()
    ds.majority.feature.res <- mclapply(as.data.frame(t(ds.majority.feature)), majority_vote, mc.cores = 8)
    ds.majority.feature.res <- as.vector(unlist(ds.majority.feature.res))
    ds.after.imp.method[,i.name] <- ds.majority.feature.res
  }
}
# fix data type
ds.after.imp.method <- ds.after.imp.method %>% as.data.frame()
ds.after.imp.method$realrinc <- ds_with_filter_col$realrinc
ds.after.imp.method$occrecode <- ds_with_filter_col$occrecode
ds.after.imp.method$wrkstat <- ds_with_filter_col$wrkstat
ds.after.imp.method$gender <- ds_with_filter_col$gender

ds.after.imp.method <- ds.after.imp.method %>% 
  mutate(age = as.numeric(levels(age))[age]) %>%
  mutate(childs = as.numeric(levels(childs))[childs]) %>%
  mutate(prestg10 = as.numeric(levels(prestg10))[prestg10])
  
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
quantile_var <- quantile(ds.w.imputed.dummies$age, probs = seq(0, 1, length.out = 5))
ds.w.imputed.dummies$age_qtil <- cut(ds.w.imputed.dummies$age, breaks = quantile_var,
                                include.lowest = TRUE)
quantile_var2 <- quantile(ds.w.imputed.dummies$childs, probs = seq(0, 1, by=1/3))
ds.w.imputed.dummies$childs_qtil <- cut(ds.w.imputed.dummies$childs, breaks = quantile_var2,
                                     include.lowest = TRUE)
quantile_var3 <- quantile(ds.w.imputed.dummies$prestg10 , probs = seq(0, 1, by=1/4))
ds.w.imputed.dummies$prestg10_qtil <- cut(ds.w.imputed.dummies$prestg10, breaks = quantile_var3,
                                        include.lowest = TRUE)

# anova test for interactions
model.ts <- lm(realrinc ~ prestg10 + age + (prestg10 * age)+ (prestg10 * childs)+(age * childs), 
               data = ds.w.imputed.dummies)
anova(model.ts)
summary(model.ts)

ggplot(ds.w.imputed.dummies, aes(x = childs, y = log_realrinc, color = as.factor(age_qtil)) ) +
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

# fast estimation of correlation matrix
cor.filter <- select_if(ds.w.imputed.dummies, is.numeric)
cor(cor.filter, method = "spearman")


# save datasets
write.csv(ds.w.imputed.dummies, file = "data")