

## Question 10.1

# Clean workspace
rm(list=ls())
cat("\014")
graphics.off()
set.seed(123)

#install.packages("rpart")
library(rpart)
#install.packages("DAAG")
library(DAAG)
#install.packages("rpart")
library(rpart)
#install.packages("rpart.plot")
library(rpart.plot)  # plotting regression trees
#install.packages("tree")
library(tree)

uscrime <- read.csv("~/Desktop/Georgia Tech Classes/ISyE 6501/Week 7 - Advanced Regression/Homework 7/Data/uscrime.csv", sep="")
head(uscrime)

## Part A: Regression Tree

# We now use the tree() function to fit a classification tree in order to predict crime rate.

tree.uscrime =tree(Crime~., data = uscrime)
summary(tree.uscrime)
# Notice that the output of summary() indicates that only four of the variables have been used in constructing the tree which are "Po1" "Pop" "LF"  "NW".
# Deviance means here the mean squared error which is high, about 47390.

# We now plot the tree.
plot(tree.uscrime)
text(tree.uscrime ,pretty=0, cex=.75)

# Now we use the cv.tree() function to see whether pruning the tree will improve performance.
cv.uscrime=cv.tree(tree.uscrime)

plot(cv.uscrime$size ,cv.uscrime$dev ,type='b')
# gives the deviance for each K (small is better). K is the numbre of leaf in the tree.
cv.uscrime$dev  
# which size is better?
best.size <- cv.uscrime$size[which(cv.uscrime$dev==min(cv.uscrime$dev))] 
best.size
# Obviously 5 leafs would give us the best model.

# let's refit the tree model the number of leafs will be no more than best.size which is 5

cv.model.pruned = prune.tree(tree.uscrime ,best=best.size)
summary(cv.model.pruned)
plot(tree.uscrime)
text(tree.uscrime ,pretty=0, cex=.75)


# In keeping with the cross-validation results, we use the unpruned tree to make predictions as the pruned tree does not give better results.
yhat=predict(tree.uscrime)

# Plot of actual vs. predicted crime values
plot(uscrime$Crime, yhat)
abline(0,1)

# Plot the residuals

plot(uscrime$Crime, scale(yhat - uscrime$Crime))
abline(0,0)

# Let's calculate the sum squared residulals (SSR):
ssr = sum((yhat-uscrime$Crime)^2)

# Let's calculate the total sum squared errors (SST):
sst = sum((uscrime$Crime - mean(uscrime$Crime))^2)

# Let's calculate the R squared:
R_squared <- 1 - ssr/sst
R_squared
# We get an R-Squared of 73%. R-Squared is the proportion of the variance in the dependent 
# variable that is predictable from the independent variable 


# Now, let's check the model quality using cross validation. Below the squared errors for each leaf:
prune.tree(tree.uscrime)$size
prune.tree(tree.uscrime)$dev

# And we compare it to thesquared errors in cross-validation:

cv.uscrime$size
cv.uscrime$dev
# The errors are much larger. The model is not doing well. 
# We notice that model is overfitting the data. We observe that there are 7 leafs for a little bit data.
# Given that for each leaf, the estimate is the average Crime, we imply that few data points are
# used to for each leaf. It is difficult to not overfit in such regaression.

## Part B: Random Forest Model

#install.packages("randomForest")
library(randomForest)

set.seed(1)

# Based on the tree regression where 4 variables were selected, we decide to limit our random forest to the same number of varaibles.
rf.uscrime=randomForest(Crime~.,data=uscrime,mtry = 4, importance =TRUE)
rf.uscrime

# How well does this model perform?
yhat.rf = predict(rf.uscrime)
plot(yhat.rf, uscrime$Crime)
abline(0,1)
# Plot residuals

plot(uscrime$Crime, scale(yhat.rf - uscrime$Crime))
abline(0,0)
mean((yhat.rf-uscrime$Crime)^2)

# Using the importance() function, we can view the importance of each variable.
# We can see that Po1 seems to be the most important variable for predictions as the %IncMSE and IncNodePurity are the highest.
# That confirms our observation in the first part where Po1 was the primary branching variable.
# %IncMSE is the most robust and informative measure. It is the increase in mse of predictions
# (estimated with out-of-bag-CV) as a result of variable j being permuted(values randomly shuffled).
#
# IncNodePurity relates to the loss function which by best splits are chosen. 
# The loss function is mse for regression and gini-impurity for classification. 
# More useful variables achieve higher increases in node purities, that is to find a split 
# which has a high inter node 'variance' and a small intra node 'variance'

importance (rf.uscrime)
varImpPlot (rf.uscrime)

# Let's compute R-squared:
r_squared <- 1 - sum((yhat.rf-uscrime$Crime)^2)/sum((uscrime$Crime - mean(uscrime$Crime))^2)
r_squared

# Now, let's try leave-one-out cross-validation:

SSE <- 0

for (i in 1:nrow(uscrime)) {
  rd <- randomForest(Crime~., data = uscrime[-i,], mtry = 4, importance = TRUE)
  SSE = SSE + (predict(rd,newdata=uscrime[i,]) - uscrime[i,16])^2
}
1 - SSE/sum((uscrime$Crime - mean(uscrime$Crime))^2)

# So the random forest model looks better than simple regression tree. It seems that has avoided overfitting as the 
# R_squared in LOVC is abou 45% which is higher than training model one of 43%.
# As shown in the Importance() function, Po1 is the most important variable. 

## Question 10.3
#Part A: Logistic Regression Model 

# Clean workspace
rm(list=ls())
cat("\014")
graphics.off()
set.seed(123)
germancredit <- read.table("~/Desktop/Georgia Tech Classes/ISyE 6501/Week 7 - Advanced Regression/Homework 7/Data/germancredit.csv", quote="\"", comment.char="")
#View(germancredit)
colnames(germancredit) = c("chk_acct", "duration", "credit_his", "purpose", 
                            "amount", "saving_acct", "present_emp", "installment_rate", "sex", "other_debtor", 
                            "present_resid", "property", "age", "other_install", "housing", "n_credits", 
                            "job", "n_people", "telephone", "foreign", "response")
head(germancredit)
str(germancredit)
summary(germancredit)
# We must convert 1s and 2s to 0s and 1s for the response variable as the binomial 
# family of glm recognises 0 and 1 as the classfication values, 
germancredit$response = germancredit$response - 1
table(germancredit$response)

# We are going to create binary variable for each factor:



# Create training (70%) and test (30%) sets for the germancredit data.
# Use set.seed for reproducibility
#install.packages("rsample")
library(rsample)
set.seed(123)
germancredit_split <- initial_split(germancredit, prop = .7)
germancredit_train <- training(germancredit_split)
germancredit_test  <- testing(germancredit_split)

# We fit a logistic regression model using all the available variables on the train data set

glm.fit = glm(response ~.,family=binomial(link = "logit"),data=germancredit_train)
summary(glm.fit)

# "glm.fit" will consider all variables and their significance level. 
# We can use stepwise model selection to select the significant variables.
# "step()" function helps in selecting the smaller set of variables.
step(glm.fit,direction = "both")
# Now we have the final list of variables from step() function, we are re-running logistic regression. 
# Also one by one we have removed insignificant variables. 
# Below the model with lowest AIC:
# response ~ chk_acct + duration + credit_his + purpose + 
#  amount + saving_acct + present_emp + installment_rate + other_debtor + 
#  other_install + housing + telephone + foreign, family = binomial(link = "logit"), 
# data = germancredit_train)

# Then we are dropping variables based on multicollinearity.   
# Detecting Multicollinearity Using Variance Inflation Factors.
library(car)
vif(glm.fit)
# First cut of logistic variables is below. We have to ensure that all variables are significant and there is 
# no multicolliearity (select only vif <2 for each of the variable).
# After running stepwise algo and VIF, below the selected model: 
glm.fit.final = glm(formula = response ~ chk_acct + duration + saving_acct + installment_rate + other_debtor + 
                      other_install + telephone + foreign, family = binomial(link = "logit"), 
                    data = germancredit_train)
  
summary(glm.fit.final)

# In this model, we have selected categorical variables, however not all of them are significant.
# We are going to create binary variable for each significant factor and run another regression.

germancredit_train$chk_acctA13[germancredit_train$chk_acct == "A13"] <- 1
germancredit_train$chk_acctA13[germancredit_train$chk_acct != "A13"] <- 0

germancredit_train$chk_acctA14[germancredit_train$chk_acct == "A14"] <- 1
germancredit_train$chk_acctA14[germancredit_train$chk_acct != "A14"] <- 0

germancredit_train$saving_acctA63[germancredit_train$saving_acct == "A63"] <- 1
germancredit_train$saving_acctA63[germancredit_train$saving_acct != "A63"] <- 0

germancredit_train$saving_acctA65[germancredit_train$saving_acct == "A65"] <- 1
germancredit_train$saving_acctA65[germancredit_train$saving_acct != "A65"] <- 0

germancredit_train$other_debtorA103[germancredit_train$other_debtor == "A103"] <- 1
germancredit_train$other_debtorA103[germancredit_train$other_debtor != "A103"] <- 0

germancredit_train$other_installA143[germancredit_train$other_install == "A143"] <- 1
germancredit_train$other_installA143[germancredit_train$other_install != "A143"] <- 0


# Now we run another logistic regression with new binary variable:

glm.fit.new = glm(formula = response ~ chk_acctA13 + chk_acctA14 + duration + saving_acctA63 + saving_acctA65+ installment_rate + other_debtorA103 + 
  other_installA143 + telephone + foreign, family = binomial(link = "logit"), data = germancredit_train)
summary(glm.fit.new)


# We remove installment_rate as its p-value is greater than 0.05 non significant.
glm.fit.new.final = glm(formula = response ~ chk_acctA13 + chk_acctA14 + duration + saving_acctA63 + saving_acctA65 + other_debtorA103 + 
                    other_installA143 + telephone + foreign, family = binomial(link = "logit"), data = germancredit_train)
summary(glm.fit.new.final)

### Now we will proceed with validation test.
# We create binary variable for each significant on the validation data set.
# germancredit_test
germancredit_test$chk_acctA13[germancredit_test$chk_acct == "A13"] <- 1
germancredit_test$chk_acctA13[germancredit_test$chk_acct != "A13"] <- 0

germancredit_test$chk_acctA14[germancredit_test$chk_acct == "A14"] <- 1
germancredit_test$chk_acctA14[germancredit_test$chk_acct != "A14"] <- 0

germancredit_test$saving_acctA63[germancredit_test$saving_acct == "A63"] <- 1
germancredit_test$saving_acctA63[germancredit_test$saving_acct != "A63"] <- 0

germancredit_test$saving_acctA65[germancredit_test$saving_acct == "A65"] <- 1
germancredit_test$saving_acctA65[germancredit_test$saving_acct != "A65"] <- 0

germancredit_test$other_debtorA103[germancredit_test$other_debtor == "A103"] <- 1
germancredit_test$other_debtorA103[germancredit_test$other_debtor != "A103"] <- 0

germancredit_test$other_installA143[germancredit_test$other_install == "A143"] <- 1
germancredit_test$other_installA143[germancredit_test$other_install != "A143"] <- 0

# test the model

y_hat<-predict(glm.fit.new.final,germancredit_test,type = "response")
y_hat

# Built a confusion matrix.

y_hat_round <- as.integer(y_hat > 0.5)

confusin_matrix <- table(y_hat_round,germancredit_test$response)
confusin_matrix

# Check model's accuracy
accuracy_rate <- (confusin_matrix[1,1] + confusin_matrix[2,2]) / sum(confusin_matrix)
accuracy_rate
# We obtain an accuracy rate of 72%.

# Develop ROC curve to determine the quality of fit
#install.packages("pROC")
library(pROC)
roc_<-roc(germancredit_test$response,y_hat_round)
roc_
roc_$auc
# Plot the ROC curve
library(ROCR)

par(mfrow=c(1, 1))

prediction(y_hat_round, germancredit_test$response) %>%
  performance(measure = "tpr", x.measure = "fpr") %>%
  plot()


# The area under the curve is 61.11%. The receiving operating characteristic (ROC) is a 
# visual measure of classifier performance. Using the proportion of positive data points 
# that are correctly considered as positive and the proportion of negative data points that 
# are mistakenly considered as positive, we generate a graphic that shows the trade off between 
# the rate at which you can correctly predict something with the rate of incorrectly predicting 
# something. Ultimately, we???re concerned about the area under the ROC curveThis means that whenever 
# Thus, the model will correctly classify data point for 57.75% of the times.

# Part B: Determine a good threshold probability

# Because the model gives a result between 0 and 1, it requires setting a threshold 
# probability to separate between ???good??? and ???bad??? answers. In this data set, 
# they estimate that incorrectly identifying a bad customer as good, is 5 times worse 
# than incorrectly classifying a good customer as bad.

loss <- c()
for(i in 1:100)
{
  y_hat_round <- as.integer(y_hat > (i/100)) 
  
  tm <-as.matrix(table(y_hat_round,germancredit_test$response))
  
  if(nrow(tm)>1) { c1 <- tm[2,1] } else { c1 <- 0 }
  if(ncol(tm)>1) { c2 <- tm[1,2] } else { c2 <- 0 }
  loss <- c(loss, c2*5 + c1)
}

plot(c(1:100)/100,loss,xlab = "Threshold",ylab = "Loss",main = "Loss vs Threshold")

which.min(loss)/100
min(loss)

# The threshold probability corresponding to minimum expected loss 
# is 0.16 and its expected loss is 171.  

