# Code for my bank marketing predictive model project 
# Packages that will be needed for this project
install.packages("scales")
install.packages("plotly")
install.packages("R.utils")
install.packages("tidymodels")
install.packages("recipes")
install.packages("pROC")
install.packages("randomForest")
install.packages("usemodels")
install.packages("ranger")
install.packages("caret")
install.packages("vip")


# Loading packages 
library(tidyverse)
library(dplyr)
library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(plotly)
library(R.utils)
library(tidymodels)
library(recipes)
library(pROC)
library(randomForest)
library(usemodels)
library(ranger)
library(caret)
library(vip)


# Loaded in the bank_additional_full data set from the UCI Machine Learning Repository
# https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

# View data to ensure it loaded properly
View(bank_additional_full)
# View structure of bank_additional_full 
str(bank_additional_full)
########### Data Cleaning ###########
# Remove values that contain "unknown" in the job, default, housing, and loan columns 
bank_cleaned <- filter(bank_additional_full, job != "unknown")
bank_cleaned <- filter(bank_cleaned, default != "unknown")
bank_cleaned <- filter(bank_cleaned, housing != "unknown")
bank_cleaned <- filter(bank_cleaned, loan != "unknown")

# Transforming data for later processing 
# Capitalizes the first letter of each value in the month column
bank_cleaned$month <- capitalize(bank_cleaned$month)
# Assigns a month number to each month; month.abb requires a capitalized first letter
# Hence the capitalization that occurred in the previous line
bank_cleaned$month_number = match(bank_cleaned$month,month.abb)
# Creates a new column that reduces months into quarters; simpler for the model
bank_cleaned$quarter <- ceiling(as.numeric(bank_cleaned$month_number) /3)

# Prepping data for further exploration and visuals
# Additional data processing will be needed in order to prep for the 
# construction of models 
# Create a table that only contains the various jobs and their respective count
grpbyjob <- bank_cleaned %>% 
  arrange(bank_cleaned$job) %>% 
  group_by(bank_cleaned$job) %>% 
  summarise(total_count=n())

# This renamed the first column in this data frame to job
colnames(grpbyjob)[1] <- "job"

# Creating a bar chart to show the job composition of the data set
# Goal is to provide some basic insights into the data 
# Distribution of Occupation Among Contacted Customers 
grpbyjob %>% 
  mutate(Job = fct_reorder(job,total_count)) %>% 
  ggplot(aes(Job,total_count, fill = Job))+
  geom_col(fill = "dodgerblue4")+
  coord_flip()+
  theme_classic()+
  labs(title = "Distribution of Occupation Among Contacted Customers",
       x = "Occupation",
       y = "Number of Customers Contacted")+
  scale_x_discrete(labels = c("student"="Student", "retired"="Retired",
                              "unemployed"="Unemployed", "admin."="Admin",
                              "management"="Management","technician"="Technician",
                              "self-employed"="Self-Employed","housemaid"="Housemaid",
                              "entrepreneur"="Entrepreneur","services"="Services",
                              "blue-collar"="Blue-Collar"))

# Make a separate table with the jobs and outcome (y)
# Count the number of "yes" and "no" in the y column 
jobydist <- bank_cleaned %>% 
  count(job,y)
as_tibble(jobydist)
# We want "yes" and "no" to be their own columns with the jobs serving as rows
jobydist <- spread(jobydist,y,n)
View(jobydist)
# We could plot "yes" and "no" but it makes more sense to make the values a percentage
# We'll measure term deposit rate by calculating yes/(yes + no)
jobydist <- jobydist %>% 
  mutate(depositrate = yes/(yes + no))
# Plotting deposit rate according to job 
jobydist %>% 
  mutate(Job = fct_reorder(job,depositrate)) %>% 
  ggplot(aes(Job,depositrate, fill = Job))+
  geom_col()+
  scale_y_continuous(labels = scales::percent)+
  coord_flip()+
  labs(title = "Students are the Group Most Likely to Subscribe to a Term Deposit",
       x= "Occupation",
       y= "Percentage of Marketing Calls Ending in a Term Deposit")+
  theme_classic()+
  theme(legend.position = "none")+
  scale_fill_manual(values=c("student"="dodgerblue4", "retired"="forestgreen",
                             "unemployed"="forestgreen", "admin."="forestgreen",
                             "management"="forestgreen","technician"="forestgreen",
                             "self-employed"="forestgreen","housemaid"="forestgreen",
                             "entrepreneur"="forestgreen","services"="forestgreen",
                             "blue-collar"="forestgreen"))+
  scale_x_discrete(labels = c("student"="Student", "retired"="Retired",
                              "unemployed"="Unemployed", "admin."="Admin",
                              "management"="Management","technician"="Technician",
                              "self-employed"="Self-Employed","housemaid"="Housemaid",
                              "entrepreneur"="Entrepreneur","services"="Services",
                              "blue-collar"="Blue-Collar"))

# Box plots with age distribution by occupation 
bank_cleaned %>% 
  ggplot(aes(job,age, color = job))+
  geom_boxplot()+
  coord_flip()+
  theme_classic()+
  labs(title = "Age Distribution by Occupation",
       x = "Occupation",
       y = "Age",
       caption = "Dots represent outliers")+
  scale_x_discrete(labels = c("student"="Student", "retired"="Retired",
                              "unemployed"="Unemployed", "admin."="Admin",
                              "management"="Management","technician"="Technician",
                              "self-employed"="Self-Employed","housemaid"="Housemaid",
                              "entrepreneur"="Entrepreneur","services"="Services",
                              "blue-collar"="Blue-Collar"))+
  scale_color_manual(values = c("dodgerblue4", "dodgerblue4", "dodgerblue4", 
                                "dodgerblue4", "dodgerblue4", "dodgerblue4",
                                "dodgerblue4", "dodgerblue4", "dodgerblue4",
                                "dodgerblue4", "dodgerblue4"))+
  theme(legend.position = "none")

# Calculate the general mean age of the data set 
mean(bank_cleaned$age)
# Calculate the success rate for the entire data set -- yes/(yes + no)
yes_counts <- table(bank_cleaned$y)
print(yes_counts)
# The success rate is 12.8%, meaning "no"/"0" represents 87.2% of values in y
# Before we can begin modeling, we need to do a bit more data processing 
glimpse(bank_cleaned)
# Need to factorize the categorical variables in the data set 
bank_cleaned <- mutate(bank_cleaned, across(where(is.character),as.factor))
# Turn "yes" in y column into 1, "no" represented by 0
levels(bank_cleaned$y) <- 0:1
# First, we are going to construct a logistic regression 
# We are dealing with class imbalance but we won't address it in the first model
# We are going to allocate 70% of our data set to training 
# Strata sets the y column as the variable we'll be predicting 
set.seed(123)
bank_splitlr <- initial_split(bank_cleaned,
                              prop = 0.7,
                              strata = y)
bank_traininglr <- bank_splitlr %>% 
  training()
bank_testlr <- bank_splitlr %>% 
  testing
# Check the number of rows in each of the new datasets 
nrow(bank_traininglr)
nrow(bank_testlr)
# The glm engine is for logistic regression and classification is the work it will be doing 
logistic_modelbc1 <- logistic_reg() %>% 
  set_engine('glm') %>% 
  set_mode('classification')
# We don't want to test the entire data set 
# Sticking to info about the customers and economic state at the time of call
logistic_fitbc1 <- logistic_modelbc1 %>% 
  fit(y ~ age + job + housing + loan + default + euribor3m +
        cons.price.idx + cons.conf.idx, data = bank_traininglr)
logistic_fitbc1

# Predict outcome categories 
class_predslr1 <- predict(logistic_fitbc1, new_data = bank_testlr,
                         type = 'class')
# Estimated probabilities for each outcome 
prob_predslr1 <- predict(logistic_fitbc1, new_data = bank_testlr,
                        type = 'prob')
# Combine test results 
bank_resultslr1 <- bank_testlr %>% 
  select(y) %>% 
  bind_cols(class_predslr1, prob_predslr1)
bank_resultslr1
# Calculate confusion matrix 
conf_mat(bank_resultslr1, truth = y, estimate = .pred_class)
# Calculate the accuracy 
accuracy(bank_resultslr1, truth = y, estimate = .pred_class)
# Calculate the ROC AUC 
roc_auc(bank_resultslr1,
        truth = y, 
        .pred_0)
# The model is okay but not great 
# It's really good at predicting when a customer won't subscribe to a term deposit
# But it struggles to predict when customers will subscribe 
# This can likely be attributed to class imbalance 
# Now we'll make a model that attempts to address class imbalance 
set.seed(123)
bank_splitlr <- initial_split(bank_cleaned,
                              prop = 0.7,
                              strata = y)
bank_traininglr <- bank_splitlr %>% 
  training()
bank_testlr <- bank_splitlr %>% 
  testing
# We'll be doing oversampling done via upSample
set.seed(123)
bank_trainup <- upSample(x=bank_traininglr[,-ncol(bank_traininglr)],
                         y=bank_traininglr$y)
table(bank_trainup$Class)
# Running the logistic regression 
logistic_modelbc2 <- logistic_reg() %>% 
  set_engine('glm') %>% 
  set_mode('classification') 
# Changed y to Class becuase of how it was restructured during oversampling
logistic_fitbc2 <- logistic_modelbc2 %>% 
  fit(Class ~ age + job + housing + loan + default + euribor3m +
        cons.price.idx + cons.conf.idx, data = bank_trainup)
logistic_fitbc2

# Predict outcome categories 
class_predslr2 <- predict(logistic_fitbc2, new_data = bank_testlr,
                          type = 'class')
# Estimated probabilities for each outcome 
prob_predslr2 <- predict(logistic_fitbc2, new_data = bank_testlr,
                         type = 'prob')

# Combine test results 
bank_resultslr2 <- bank_testlr %>% 
  select(y) %>% 
  bind_cols(class_predslr2, prob_predslr2)
bank_resultslr2
# Calculate confusion matrix 
conf_mat(bank_resultslr2, truth = y, estimate = .pred_class)
# Calculate the accuracy 
accuracy(bank_resultslr2, truth = y, estimate = .pred_class)
# Calculate the ROC AUC 
roc_auc(bank_resultslr2,
        truth = y, 
        .pred_0)

# Although we see a dip in accuracy after oversampling, we see a significant
# increase in sensitivity, which means we're getting better at predicting successes

# Now that we've made logistic regression models, it's time to make a random 
# forest model. We'll do one with the original sample and one with oversampling

# Need to split up data 
set.seed(234)
rf_split1 <- initial_split(bank_cleaned, strata = y)
rf_train1 <- training(rf_split1)
rf_test1 <- testing(rf_split1)

# Making resamples 
# Will be doing resamplign to optimize model performance 
set.seed(234)
bank_folds1 <- bootstraps(rf_train1, strata = y)
bank_folds1

# Building the model 
# This line spits out code for best practices in making a model 
# Can copy and paste code in output as a skeleton for what the tidymodel should
# look like 
use_ranger(y ~., data = rf_train1)

ranger_recipe1 <- 
  recipe(formula = y ~ age + job + euribor3m +
           cons.price.idx + cons.conf.idx, data = rf_train1) 

ranger_spec1 <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 500) %>% 
  set_mode("classification") %>% 
  set_engine("ranger") 

ranger_workflow1 <- 
  workflow() %>% 
  add_recipe(ranger_recipe1) %>% 
  add_model(ranger_spec1) 

set.seed(60579)
doParallel::registerDoParallel()
ranger_tune1 <-
  tune_grid(ranger_workflow1, resamples = bank_folds1,
            grid = 8)
# Picks the best model 
show_best(ranger_tune1)
autoplot(ranger_tune1)

# Finalizes the workflow for this model 
final_rf1 <- ranger_workflow1 %>% 
  finalize_workflow(select_best(ranger_tune1))
final_rf1

bank_fit1 <- last_fit(final_rf1,rf_split1)
bank_fit1

# Gives accuracy and AUC; I need to figure out how to get nore values
collect_metrics(bank_fit1)

# Predictions
collect_predictions(bank_fit1)

# Produces a bar chart with the importance of the variables 
imp_spec1 <-ranger_spec1 %>% 
  finalize_model(select_best(ranger_tune1)) %>% 
  set_engine("ranger", importance = "permutation")

workflow() %>% 
  add_recipe(ranger_recipe1) %>% 
  add_model(imp_spec1) %>% 
  fit(rf_train1) %>% 
  pull_workflow_fit() %>% 
  vip(aesthetics = list(fill = "dodgerblue4"))+
  ggtitle("The Euribor3m is the Most Important Predictor Variable")+
  xlab("Predictor Variables")

# Collect predictions to create a confusion matrix
predictions1 <- collect_predictions(bank_fit1)
true_labels1 <- predictions1$y
predicted_labels1 <- predictions1$.pred_class

confusion_matrixrf1 <- confusionMatrix(predicted_labels1, true_labels1)
confusion_matrixrf1

# This model performed better than the first logistic regression model in 
# predicting true positives (which is our goal)
# Now, let's make a random forest model that accounts for class imbalance 
# Set up the splits; we'll oversample the training split in a second
set.seed(345)
rf_split2 <- initial_split(bank_cleaned, strata = y)
rf_train2 <- training(rf_split2)
rf_test2 <- testing(rf_split2)

set.seed(345)
rf_trainup <- upSample(x=rf_train2[,-ncol(bank_traininglr)],
                         y=rf_train2$y)
# upSample turned y into Class but we need it to be named y for the sake 
# of compatibility when aligning the data set with rf_split2 later
rf_trainup <- rename(rf_trainup, y = Class)
table(rf_trainup$y)
# After running that, we can see that there are 19306 values of "yes" and "no"

# Will be doing resamplign to optimize model performance 
set.seed(345)
bank_folds2 <- bootstraps(rf_trainup, strata = y)
bank_folds2

# Model
# upSample makes our y value register as Class so we'll change it in the formula
ranger_recipe2 <- 
  recipe(formula = y ~ age + job + euribor3m +
           cons.price.idx + cons.conf.idx, data = rf_trainup) 

ranger_spec2 <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 500) %>% 
  set_mode("classification") %>% 
  set_engine("ranger") 

ranger_workflow2 <- 
  workflow() %>% 
  add_recipe(ranger_recipe2) %>% 
  add_model(ranger_spec2) 

set.seed(60579)
doParallel::registerDoParallel()
ranger_tune2 <-
  tune_grid(ranger_workflow2, resamples = bank_folds2,
            grid = 8)
# Picks the best model 
show_best(ranger_tune2)
autoplot(ranger_tune2)
# Finalizes model 
final_rf2 <- ranger_workflow2 %>% 
  finalize_workflow(select_best(ranger_tune2))
final_rf2

bank_fit2 <- last_fit(final_rf2,rf_split2)
bank_fit2

# Gives accuracy and AUC; I need to figure out how to get nore values
collect_metrics(bank_fit2)

# Predictions
collect_predictions(bank_fit2)

# Produces a bar chart with the importance of the variables 
imp_spec2 <-ranger_spec2 %>% 
  finalize_model(select_best(ranger_tune2)) %>% 
  set_engine("ranger", importance = "permutation")

workflow() %>% 
  add_recipe(ranger_recipe2) %>% 
  add_model(imp_spec2) %>% 
  fit(rf_trainup) %>% 
  pull_workflow_fit() %>% 
  vip(aesthetics = list(fill = "dodgerblue4"))

# Collect predictions to create a confusion matrix
predictions2 <- collect_predictions(bank_fit2)
true_labels2 <- predictions2$y
predicted_labels2 <- predictions2$.pred_class

confusion_matrixrf2 <- confusionMatrix(predicted_labels2, true_labels2)
confusion_matrixrf2

# Adjusting the sampling did increase the number of accurately predicted 
# subscriptions, but the results pale in comparison to those of the 
# second logistic regression model





