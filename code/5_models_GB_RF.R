# set wd
tryCatch({
  setwd("C:/Users/lbergmann/OneDrive - RWI–Leibniz-Institut für Wirtschaftsforschung e.V/Dokumente/Promotion/StatisticalLearning/gender_pay_gap_project_1")
}, error = function(err1) {
  # If an error occurs, handle it by setting the working directory to the second path
  setwd("/home/user/Downloads/info_tud/statistical_learning_UDE/gender_pay_gap_project")
})
# import functions file
source('code/utils_functions.R')

# packages
library(caret)
library(dplyr)
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

# load data
ds <- fread("data/full_data_w_dummies_interaction.csv")
ds$wrkstat <- as.factor(ds$wrkstat)
ds$educcat <- as.factor(ds$educcat)
ds$gender <- as.factor(ds$gender)
ds$occrecode <- as.factor(ds$occrecode)
ds$maritalcat <- as.factor(ds$maritalcat)

# split in train, validation and test data
set.seed(123)

trainindex_filter <- createDataPartition(y = ds$realrinc, p = 0.7, list = FALSE) # Split the data into training (70%) and remaining (30%)
train_filter <- ds[trainindex_filter, ]
remaining_filter <- ds[-trainindex_filter, ]

cat("training dim: ", nrow(train_filter), ", test dim: ", nrow(remaining_filter), "\n")

full.col.names <- colnames(ds)
occrecode.labels <- full.col.names[match("armed_forces", full.col.names):match("transportation", full.col.names)]
occrecode.labels <- occrecode.labels[occrecode.labels != "armed_forces"]
educcat.labels <- full.col.names[match("bachelor", full.col.names):match("less_than_high_school", full.col.names)]
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

x_cols <- c(colnames(ds)[match("age", colnames(ds)):match("childs", colnames(ds))],
            "inter_age_prestg","inter_age_childs", "inter_prestg_childs",
            "age_sqr", "wrkstat","educcat","gender", "occrecode", "maritalcat", x_cols_inter)

x_cols_wo_inter <- c(colnames(ds)[match("age", colnames(ds)):match("childs", colnames(ds))],
                     "inter_age_prestg","inter_age_childs", "inter_prestg_childs",
                     "age_sqr", "wrkstat","educcat","gender", "occrecode", "maritalcat")
ytarget <- "realrinc"
ytarget.log <- "log_realrinc"

x_cols_dummys <- c( colnames(ds)[match("age", colnames(ds)):match("childs", colnames(ds))],
                    colnames(ds)[match("armed_forces", colnames(ds)):match("transportation", colnames(ds))],
                    colnames(ds)[match("full_time", colnames(ds)):match("unemployed_laid_off" , colnames(ds))],
                    colnames(ds)[match("bachelor", colnames(ds)):match("less_than_high_school" , colnames(ds))],
                    colnames(ds)[match("divorced", colnames(ds)):match("widowed", colnames(ds))],
                    "female", "age_sqr",
                    "inter_age_prestg","inter_age_childs", "inter_prestg_childs", x_cols_inter) #"male_high_income"
x_cols_dummys <- x_cols_dummys[!x_cols_dummys %in% c("armed_forces", "other_wrkstat", "junior_college", "separated")]
x_cols_dummys_wo_inter <- x_cols_dummys[!x_cols_dummys %in% x_cols_inter]

# ******************************************************************************
# Convert the data to a Task object, define task
task <- TaskRegr$new(id = "gap_task",
                     backend = ( train_filter %>% 
                                   select(all_of(c(x_cols,ytarget))) ),
                     target = ytarget)

task.wdummyes <- TaskRegr$new(id = "gap_task",
                              backend = ( train_filter %>% 
                                            select(all_of(c(x_cols_dummys,ytarget))) ),
                              target = ytarget)

task.wdummyes.wlog <- TaskRegr$new(id = "gap_task",
                                   backend = ( train_filter %>% 
                                                 select(all_of(c(x_cols_dummys,ytarget.log))) ),
                                   target = ytarget.log )

task.wlog <- TaskRegr$new(id = "gap_task",
                          backend = ( train_filter %>% 
                                        select(all_of(c(x_cols,ytarget.log))) ),
                          target = ytarget.log)

task.wo <- TaskRegr$new(id = "gap_task",
                        backend = ( train_filter %>% 
                                      select(all_of(c(x_cols_wo_inter,ytarget))) ),
                        target = ytarget)

task.wo.wlog <- TaskRegr$new(id = "gap_task",
                             backend = ( train_filter %>% 
                                           select(all_of(c(x_cols_wo_inter,ytarget.log))) ),
                             target = ytarget.log)

# define tuning type, batch_size: # of config./combinations per batch
tnr_rdgrid_search = tnr("random_search", batch_size = 10)

# define cross-validation and error metric
rsmp_cv3 = rsmp("cv", folds = 3)
msr_ce = msr("regr.rmse")

# ------------------------------------------------------------------------------
# tuning rf
learner = lrn("regr.lightgbm",
              boosting = "rf",
              objective = "regression",
              max_depth = to_tune(seq(8,12,1)),
              num_leaves = to_tune(seq(20,90,1)), 
              max_bin = to_tune(seq(20,35,1)), 
              num_iterations  = to_tune(seq(60,95,1)),
              min_data_in_leaf = to_tune(seq(20,40,1)),
              min_data_in_bin = to_tune(seq(10,25, 1)), 
              feature_fraction_bynode = to_tune(seq(0.6,0.9,0.1)), 
              bagging_fraction = to_tune(seq(0.8,0.9,0.1)),
              bagging_freq = to_tune(seq(3,5,1)),
              feature_fraction = to_tune(seq(0.7,0.9,0.1)),
              convert_categorical = TRUE,
              force_col_wise = TRUE,
              verbose = 1,
              num_threads = 5,
              seed = 123
)

# begin training process, with 200 configurations
init_time <- Sys.time()
instance.rf = tune(
  tuner = tnr_rdgrid_search,
  task = task,
  learner = learner,
  resampling = rsmp_cv3,
  measures = msr_ce,
  term_evals = 200,
  store_models = FALSE
)
cat('finish rf.....\n')
end_time <- Sys.time()
print(difftime(init_time,end_time))

comb.best.model.rf <- (instance.rf$result)
result.table.rf <- (as.data.table(instance.rf$archive, measures = msrs(c("regr.mse","regr.mae")) ))

# fit final model on complete data set with the best combination
lrn_rf_tuned = lrn("regr.lightgbm")
lrn_rf_tuned$param_set$values = instance.rf$result_learner_param_vals
lrn_rf_tuned$train(task)

# prediction over test data set
y.predict_rf <- predict(lrn_rf_tuned$model,
                        remaining_filter %>% 
                          select(all_of(x_cols)) %>% 
                          as.matrix() 
)

rmse.test_rf <- sqrt(mean(( as.vector(as.matrix(remaining_filter %>% select(ytarget) ) ) - y.predict_rf)^2)) 

# ------------------------------------------------------------------------------
# tuning rf with log transform
learner.wlog = lrn("regr.lightgbm",
                   boosting = "rf",
                   objective = "regression",
                   max_depth = to_tune(seq(7,12,1)), 
                   num_leaves = to_tune(seq(60,120,1)),  
                   max_bin = to_tune(seq(40,95,1)),
                   num_iterations  = to_tune(seq(70,125,1)), 
                   min_data_in_leaf = to_tune(seq(20,40,1)), 
                   min_data_in_bin = to_tune(seq(20,50, 1)), 
                   feature_fraction_bynode = to_tune(seq(0.4,0.7,0.1)),
                   bagging_fraction = to_tune(seq(0.6,0.9,0.1)),
                   bagging_freq = to_tune(seq(3,12,1)),
                   feature_fraction = to_tune(seq(0.8,0.9,0.1)),
                   convert_categorical = TRUE,
                   force_col_wise = TRUE,
                   verbose = 1,
                   num_threads = 5,
                   seed = 123
)


# begin training process, with 200 configurations
init_time <- Sys.time()
instance.rf.wlog = tune(
  tuner = tnr_rdgrid_search,
  task = task.wlog,
  learner = learner.wlog,
  resampling = rsmp_cv3,
  measures = msr_ce,
  term_evals = 20,
  store_models = FALSE
)
cat('finish gb.....\n')
end_time <- Sys.time()
print(difftime(init_time,end_time))

comb.best.model.rf.wlog <- (instance.rf.wlog$result)
result.table.rf.wlog <- (as.data.table(instance.rf.wlog$archive, measures = msrs(c("regr.mse","regr.mae")) ))

# fit final model on complete data set with the best combination
lrn_rf_tuned.wlog = lrn("regr.lightgbm")
lrn_rf_tuned.wlog$param_set$values = instance.rf.wlog$result_learner_param_vals
lrn_rf_tuned.wlog$train(task.wlog)

# prediction over test data set
y.predict_rf.wlog <- predict(lrn_rf_tuned.wlog$model,
                             remaining_filter %>% 
                               select(all_of(x_cols)) %>% 
                               as.matrix() 
)

rmse.test_rf.wlog <- sqrt(mean(( as.vector(as.matrix(remaining_filter %>% select(ytarget) ) ) - exp(y.predict_rf.wlog) )^2)) 

# ******************************************************************************
# GB experiments
# establish hyper parameters GB

learner.gb = lrn("regr.lightgbm",
                 boosting = "gbdt",
                 objective = "regression",
                 max_depth = to_tune(seq(5, 10, 1)),
                 max_bin = to_tune(seq(25,50,1)),
                 num_leaves = to_tune(seq(6,10,1)), 
                 min_data_in_leaf = to_tune(seq(40,60,1)),
                 min_data_in_bin = to_tune(seq(20,35,1)),
                 feature_fraction = to_tune(seq(0.3,0.6,0.1)), 
                 feature_fraction_bynode = to_tune(seq(0.1,0.4,0.1)),
                 learning_rate = to_tune(seq(0.02, 0.05, 0.01)), 
                 num_iterations  = to_tune(seq(35,45,1)), 
                 lambda_l1 = to_tune(seq(0.35, 0.45, 0.01)), 
                 lambda_l2 = to_tune(seq(0.29, 0.39, 0.01)), 
                 convert_categorical = TRUE,
                 force_col_wise = TRUE,
                 verbose = 1,
                 num_threads = 5,
                 seed = 123
)

# begin training process, with 100 configurations
init_time <- Sys.time()
instance.gb = tune(
  tuner = tnr_rdgrid_search,
  task = task,
  learner = learner.gb,
  resampling = rsmp_cv3,
  measures = msr_ce,
  term_evals = 200,
  store_models = FALSE
)
cat('finish gb.....\n')
end_time <- Sys.time()
print(difftime(init_time,end_time))

comb.best.model.gb <- (instance.gb$result)
result.table.gb <- (as.data.table(instance.gb$archive, measures = msrs(c("regr.mse","regr.mae")) ))

# fit final model on complete data set with the best combination
lrn_gb_tuned = lrn("regr.lightgbm")
lrn_gb_tuned$param_set$values = instance.gb$result_learner_param_vals
lrn_gb_tuned$train(task)

# prediction over test data set
y.predict_gb <- predict(lrn_gb_tuned$model, 
                        remaining_filter %>% 
                          select(all_of(x_cols)) %>% 
                          as.matrix() 
)
rmse.test_gb <- sqrt(mean(( as.vector(as.matrix(remaining_filter %>% select(ytarget) ) ) - y.predict_gb)^2))

# ******************************************************************************
# GB experiments with log transformation
# establish hyper parameters GB
learner.gb.wlog = lrn("regr.lightgbm",
                      boosting = "gbdt",
                      objective = "regression",
                      max_depth = to_tune(seq(5, 12, 1)),
                      max_bin = to_tune(seq(25,60,1)), 
                      num_leaves = to_tune(seq(6,60,1)), 
                      min_data_in_leaf = to_tune(seq(25,70,1)),
                      min_data_in_bin = to_tune(seq(15,50,1)),
                      feature_fraction = to_tune(seq(0.1,0.9,0.1)), 
                      feature_fraction_bynode = to_tune(seq(0.5,0.9,0.1)),
                      learning_rate = to_tune(seq(0.01, 0.2, 0.01)), 
                      num_iterations  = to_tune(seq(35,80,1)), 
                      lambda_l1 = to_tune(seq(0.01, 0.5, 0.01)),
                      lambda_l2 = to_tune(seq(0.01, 0.5, 0.01)),
                      convert_categorical = TRUE,
                      force_col_wise = TRUE,
                      verbose = 1,
                      num_threads = 5,
                      seed = 123
)

# begin training process, with 100 configurations
init_time <- Sys.time()
instance.gb.wlog = tune(
  tuner = tnr_rdgrid_search,
  task = task.wlog,
  learner = learner.gb.wlog,
  resampling = rsmp_cv3,
  measures = msr_ce,
  term_evals = 250,
  store_models = FALSE
)
cat('finish gb.....\n')
end_time <- Sys.time()
print(difftime(init_time,end_time))

comb.best.model.gb.wlog <- (instance.gb.wlog$result)
result.table.gb.wlog <- (as.data.table(instance.gb.wlog$archive, measures = msrs(c("regr.mse","regr.mae")) ))

# fit final model on complete data set with the best combination
lrn_gb_tuned.wlog = lrn("regr.lightgbm")
lrn_gb_tuned.wlog$param_set$values = instance.gb.wlog$result_learner_param_vals
lrn_gb_tuned.wlog$train(task.wlog)

# prediction over test data set
y.predict_gb.wlog <- predict(lrn_gb_tuned.wlog$model, 
                             remaining_filter %>% 
                               select(all_of(x_cols)) %>% 
                               as.matrix() 
)
rmse.test_gb.wlog <- sqrt(mean(( as.vector(as.matrix(remaining_filter %>% select(ytarget) ) ) - exp(y.predict_gb.wlog) )^2)) 


# ------------------------------------------------------------------------------
# experiments GAM --------------------------------------------------------------
# ------------------------------------------------------------------------------
form.str.test <- paste("log_realrinc ~ s(age, bs = 'cr', k = 9) + s(prestg10, bs = 'cr', k=9) + ti(age,prestg10) + ti(age,childs) +",
                       paste( x_cols_dummys, collapse = " + ")
)
form.format.test <- as.formula(form.str.test)
learner.gam.wlog.smooth.tensor = lrn("regr.gam",
                                     family = 'gaussian',
                                     select = TRUE,
                                     gamma = to_tune(seq(1,3,0.5)),
                                     formula = form.format.test
)


init_time <- Sys.time()
instance.gam.wlog.sm.ti = tune(
  tuner = tnr_rdgrid_search,
  task = task.wdummyes.wlog,
  learner = learner.gam.wlog.smooth.tensor,
  resampling = rsmp_cv3,
  measures = msr_ce,
  term_evals = 1,
  store_models = FALSE
)
cat('finish dart.....\n')
end_time <- Sys.time()
print(difftime(init_time,end_time))

comb.best.model.gam.wlog.sm.ti <- (instance.gam.wlog.sm.ti$result)
result.table.gam.wlog.sm.ti <- (as.data.table(instance.gam.wlog.sm.ti$archive, measures = msrs(c("regr.mse","regr.mae")) ))

# fit final model on complete data set with the best combination
lrn_gam_tuned.wlog.sm.ti = lrn("regr.gam")
lrn_gam_tuned.wlog.sm.ti$param_set$values = instance.gam.wlog.sm.ti$result_learner_param_vals
lrn_gam_tuned.wlog.sm.ti$train(task.wdummyes.wlog)

# prediction over test data set
y.predict_gam.wlog.sm.ti <- predict(lrn_gam_tuned.wlog.sm.ti, 
                                    newdata = remaining_filter %>% 
                                      select(all_of(x_cols_dummys))
)

rmse.test_gam.wlog.sm.ti <- sqrt(mean(( as.vector(as.matrix(remaining_filter %>% select(ytarget) ) ) - exp(y.predict_gam.wlog.sm.ti) )^2)) 
