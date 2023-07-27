# set wd
setwd("/home/user/Downloads/info_tud/statistical_learning_UDE/gender_pay_gap_project")
# setwd("C:/Users/lbergmann/OneDrive - RWI–Leibniz-Institut für Wirtschaftsforschung e.V/Dokumente/Promotion/StatisticalLearning/Project")

# packages
library(dplyr)
library(GGally)

# import ds
ds <- read.csv('data/gss_wages_train.csv')
summary(ds)

cat("the dataset has ", nrow(ds), 'rows, and ', round(sum(is.na(ds$realrinc))/nrow(ds)*100,2), "% of missing values for the outcome \n")


###create dummies
###test 1
#occupation
#ds_occ1 <- ds %>% distinct(occ10) #there are 537 different occupations
ds_occ2 <- ds %>% distinct(occrecode)#there are 12 different occupations: Office, Professional, NA, Business/Finance, Construction, Sales, Transportation, Service, Production, Farming, Installation, Armed Forces
ds$Office <-  ifelse(ds$occrecode == "Office and Administrative Support", 1, 0)
ds$Office <- ifelse(is.na(ds$occrecode) | ds$Office == 0, 0, 1)
ds$Professional <-  ifelse(ds$occrecode == "Professional", 1, 0)
ds$Professional <- ifelse(is.na(ds$occrecode) | ds$Professional == 0, 0, 1)
ds$Business <- ifelse(ds$occrecode == "Business/Finance", 1, 0)
ds$Business <- ifelse(is.na(ds$occrecode) | ds$Business == 0, 0, 1)
ds$Construction  <- ifelse(ds$occrecode == "Construction/Extraction", 1, 0)
ds$Construction <- ifelse(is.na(ds$occrecode) | ds$Construction == 0, 0, 1)
ds$Sales <- ifelse(ds$occrecode == "Sales", 1, 0)
ds$Sales <- ifelse(is.na(ds$occrecode) | ds$Sales == 0, 0, 1)
ds$Transportation <- ifelse(ds$occrecode == "Transportation", 1, 0)
ds$Transportation <- ifelse(is.na(ds$occrecode) | ds$Transportation == 0, 0, 1)
ds$Service <- ifelse(ds$occrecode == "Service", 1, 0)
ds$Service <- ifelse(is.na(ds$occrecode) | ds$Service == 0, 0, 1)
ds$Production <- ifelse(ds$occrecode == "Production", 1, 0)
ds$Production <- ifelse(is.na(ds$occrecode) | ds$Production == 0, 0, 1)
ds$Farming <- ifelse(ds$occrecode == "Farming, Fishing, and Forestry", 1, 0)
ds$Farming <- ifelse(is.na(ds$occrecode) | ds$Farming == 0, 0, 1)
ds$Installation <- ifelse(ds$occrecode == "Installation, Maintenance, and Repair", 1, 0)
ds$Installation <- ifelse(is.na(ds$occrecode) | ds$Installation == 0, 0, 1)
ds$Army <- ifelse(ds$occrecode == "Armed Forces", 1, 0)
ds$Army <- ifelse(is.na(ds$occrecode) | ds$Army == 0, 0, 1)
ds$WhiteCollar <- ifelse(ds$Office == 1 |ds$Professional == 1 |ds$Business == 1 |ds$Sales ==1, 1,0)
summary(ds$WhiteCollar)

#work status
#ds_lfs <- ds %>% distinct(wrkstat)
ds$school <- ifelse(ds$wrkstat == "School",1,0)
ds$school <- ifelse(is.na(ds$wrkstat) | ds$school == 0, 0,1)
ds$fulltime <- ifelse(ds$wrkstat == "Full-Time",1,0)
ds$fulltime <- ifelse(is.na(ds$wrkstat) | ds$fulltime == 0, 0,1)
ds$housekeeper <- ifelse(ds$wrkstat == "Housekeeper",1,0)
ds$housekeeper <- ifelse(is.na(ds$wrkstat) | ds$housekeeper == 0, 0,1)
ds$retired <- ifelse(ds$wrkstat == "Retired",1,0)
ds$retired <- ifelse(is.na(ds$wrkstat) | ds$retired == 0, 0,1)
ds$parttime <- ifelse(ds$wrkstat == "Part-Time",1,0)
ds$parttime <- ifelse(is.na(ds$wrkstat) | ds$parttime == 0, 0,1)
ds$unemployed <- ifelse(ds$wrkstat == "Unemployed, Laid Off" | ds$wrkstat == "Temporarily Not Working",1,0)
ds$unemployed <- ifelse(is.na(ds$wrkstat) | ds$unemployed == 0, 0,1)

#education
#ds_education <- ds %>% distinct(educcat)
ds$highschool <- ifelse(ds$educcat == "High School",1,0)
ds$highschool <- ifelse(is.na(ds$educcat) | ds$highschool == 0, 0,1)
ds$bachelor <- ifelse(ds$educcat == "Bachelor",1,0)
ds$bachelor <- ifelse(is.na(ds$educcat) | ds$bachelor == 0, 0,1)
ds$nodeg <- ifelse(ds$educcat == "Less Than High School",1,0)
ds$nodeg <- ifelse(is.na(ds$educcat) | ds$nodeg == 0, 0,1)
ds$graduate <- ifelse(ds$educcat == "Graduate",1,0)
ds$graduate <- ifelse(is.na(ds$educcat) | ds$graduate == 0, 0,1)
ds$juncollege <- ifelse(ds$educcat == "Junior College",1,0)
ds$juncollege <- ifelse(is.na(ds$educcat) | ds$juncollege == 0, 0,1)

#marital status
#ds_mar <- ds %>% distinct(maritalcat)
ds$married <- ifelse(ds$maritalcat == "Married",1,0)
ds$married <- ifelse(is.na(ds$maritalcat) | ds$married == 0,0,1)
ds$widowed <- ifelse(ds$maritalcat == "Widowed",1,0)
ds$widowed <- ifelse(is.na(ds$maritalcat) | ds$widowed == 0,0,1)
ds$divorced <- ifelse(ds$maritalcat == "Divorced" | ds$maritalcat == "Separated",1,0)
ds$divorced <- ifelse(is.na(ds$maritalcat) | ds$divorced == 0,0,1)
ds$unmarried <- ifelse(ds$maritalcat == "Never Married",1,0)
ds$unmarried <- ifelse(is.na(ds$maritalcat) | ds$unmarried == 0,0,1)

#gender
ds$female <- ifelse(ds$gender == "Female", 1, 0)


#dataset with realincr = 0 for housekeepers
ds_na <- ds
ds_na$realrinc <- ifelse(ds$housekeeper == 1 & is.na(ds$realrinc), 0, ds$realrinc)
summary(ds$realrinc)
summary(ds_na$realrinc)
ds_na <- ds_na[!is.na(ds$realrinc),]

# filtered dataset, not using obs, with missing value for outcome
ds_filter <- ds[!is.na(ds$realrinc),]
summary(ds_filter)

cat("the dataset has ", nrow(ds_filter), 'rows \n')
summary(ds_filter)

# fast estimation of correlation matrix
cor.filter <- select_if(ds_filter, is.numeric)
cor.filter <- na.omit(cor.filter)
cor(cor.filter, method = "spearman")

# pair plots
pairs(cor.filter)
ggpairs(cor.filter)

#save datasets
save(ds, file = "data/ds.Rda")
save(ds_na, file = "data/ds_na.Rda")
save(ds_filter, file = "data/ds_filter.Rda")
