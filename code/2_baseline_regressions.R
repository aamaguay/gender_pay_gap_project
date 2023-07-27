# set wd
setwd("/home/user/Downloads/info_tud/statistical_learning_UDE/gender_pay_gap_project")
setwd("C:/Users/lbergmann/OneDrive - RWI–Leibniz-Institut für Wirtschaftsforschung e.V/Dokumente/Promotion/StatisticalLearning/gender_pay_gap_project_1")

# packages
library(dplyr)
library(GGally)
library(caret)

#load data
ds <- read.csv("data/full_data_w_dummies_interaction.csv")

#split in train, validation and test data
set.seed(123)

trainindex_filter <- createDataPartition(y = ds_filter$realrinc, p = 0.7, list = FALSE) # Split the data into training (70%) and remaining (30%)
train_filter <- ds_filter[trainindex_filter, ]
remaining_filter <- ds_filter[-trainindex_filter, ]

valid_test_index_filter <- createDataPartition(y = remaining_filter$realrinc, p = 0.5, list = FALSE) # Split the remaining data into validation (50%) and test (50%)
validation_filter <- remaining_filter[valid_test_index_filter, ]
test_filter <- remaining_filter[-valid_test_index_filter, ]


#OLS regressions
mod_full <- lm(realrinc ~ . - occrecode -wrkstat - gender -educcat -maritalcat, data = ds)
summary(mod_full)

mod_wo_interactions <- lm(realrinc ~ age + prestg10 + childs + armed_forces + business_finance + construction_extraction + farming_fishing_and_forestry + 
                            installation_maintenance_and_repair + office_and_administrative_support + production + professional + sales + service + 
                             + full_time + housekeeper + part_time + retired + school + temporarily_not_working + unemployed_laid_off + bachelor + 
                            graduate + highschool + juniorcollege +  + divorced + married + nevermarried + separated  + female, data = ds)
summary(mod_wo_interactions)

mod_loginc <- lm(log_realrinc ~ age + prestg10 + childs + armed_forces + business_finance + construction_extraction + farming_fishing_and_forestry + 
                   installation_maintenance_and_repair + office_and_administrative_support + production + professional + sales + service + 
                   + full_time + housekeeper + part_time + retired + school + temporarily_not_working + unemployed_laid_off + bachelor + 
                   graduate + highschool + juniorcollege + divorced + married + nevermarried + separated  + female, data = ds)
summary(mod_loginc)

mod_age2 <- lm(realrinc ~ age + age_sqr + prestg10 + childs + armed_forces + business_finance + construction_extraction + farming_fishing_and_forestry + 
                 installation_maintenance_and_repair + office_and_administrative_support + production + professional + sales + service + 
                 + full_time + housekeeper + part_time + retired + school + temporarily_not_working + unemployed_laid_off + bachelor + 
                 graduate + highschool + juniorcollege +  divorced + married + nevermarried + separated  + female, data = ds)
summary(mod_age2)


m2 <- with(ds,lm(realrinc ~ . - occrecode -wrkstat - gender -educcat -maritalcat)) ### this does not work for me
summary(pool(m2))
