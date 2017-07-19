library(utils)
library(caret)
library(scales)
library(Amelia)
library(xtable)
library(ggplot2)
library(gridExtra)
library(data.table)
library(randomForest)

setwd("~/Desktop/projects/dk_project")

####-----------------------------------------------
####----------- Part 1: Read clean and explore data
####-----------------------------------------------

## Read data, set column names and remove duplicates
train <- data.table(read.csv("us_census_full/census_income_learn.csv", header=FALSE))
test <- data.table(read.csv("us_census_full/census_income_test.csv", header=FALSE))
col_headings <- read.csv("us_census_full/col_headings.csv", header=FALSE)
col_headings <- as.character(col_headings$V1)
names(train) <- col_headings; names(test) <- col_headings
train <- unique(train); test <- unique(test)

## First check distribution of classes, visualize distributions of variables
sapply(train, function(x) sum(is.na(x)))  # check for NA values
table(train$salary)
print(xtable(table(train$salary), caption="Class Distribution", label="class_dist"), include.rownames=FALSE)

# Numeric variables distributions and remove possible outliers
g <- ggplot(data=melt(train), aes(x=value)) + geom_histogram(color='firebrick', fill='lightblue') 
g + facet_wrap(~variable, scales='free_x')  # numeric
ggplot(data=train[wage_per_hour>0], aes(x=wage_per_hour)) + geom_density(color='firebrick', fill='lightblue')
ggplot(data=train[divdends_from_stocks>0], aes(x=divdends_from_stocks)) + geom_density(color='firebrick', fill='lightblue')
ggplot(data=train[capital_losses>0], aes(x=capital_losses)) + geom_density(color='firebrick', fill='lightblue')
ggplot(data=train[capital_gains>0], aes(x=capital_gains)) + geom_density(color='firebrick', fill='lightblue')
train <- train[wage_per_hour <= 5000]
train <- train[divdends_from_stocks <= 25000]

# Factor variables distributions
factor_data <- train
f <- sapply(factor_data, is.factor)
factor_data[f] <- lapply(factor_data[f], as.character)
factor_data <- subset(factor_data, select=-c(age, wage_per_hour, capital_gains, capital_losses, divdends_from_stocks,
                                             veterans_benefits, weeks_worked_in_year, year))
ggplot(data=melt(factor_data), aes(x=value)) + geom_bar(color='firebrick', fill='lightblue') + facet_wrap(~variable, scales='free_x')


## Convert some numeric to factor
train$num_persons_worked_for_employer <- factor(train$num_persons_worked_for_employer, 
                                                levels=unique(train$num_persons_worked_for_employer))
train$own_business_or_self_employed <- factor(train$own_business_or_self_employed, 
                                              levels=unique(train$own_business_or_self_employed))
train$veterans_benefits <- factor(train$veterans_benefits, levels=unique(train$veterans_benefits))
train$year <- factor(train$year, levels=unique(train$year))
# weeks_worked_in_year is generally either 0 or 52 but leave this as numeric for now

## Simplify Factor variables
degrees <- c(" Masters degree(MA MS MEng MEd MSW MBA)", " Prof school degree (MD DDS DVM LLB JD)", " High school graduate",
             " Doctorate degree(PhD EdD)", " Bachelors degree(BA AB BS)", " Associates degree-academic program")
train[!(education %in% degrees)]$education <- "School"
train[education %in% c(" Prof school degree (MD DDS DVM LLB JD)"," Doctorate degree(PhD EdD)")]$education <- "Doctorate"
train[education %in% c(" Bachelors degree(BA AB BS)", " Associates degree-academic program")]$education <- "Bachelors"

train[grepl("householder", train$state_of_previous_residence)]$state_of_previous_residence <- "Householder"
train[!(grepl("householder", train$state_of_previous_residence))]$state_of_previous_residence <- "Not Householder"


####--------------------------------------------------
####----------- Part 2: Prepare Data for Random Forest
####--------------------------------------------------

## Set aside a validation set
set.seed(1234)
train_ind <- sample(seq_len(nrow(train)), size=floor(0.75 * nrow(train)))
train_set <- train[train_ind]
val_set <- train[-train_ind]

## Resample from minority class and sub-sample from majority class to balance data
minority <- train_set[salary== " 50000+."]
majority <- train_set[salary== " - 50000."]
smp_ind <- sample(seq_len(nrow(majority)), size=floor(0.5 * nrow(majority)))
majority <- majority[smp_ind]
balanced_train_set <- rbind(majority, minority, minority, minority, minority, minority)
balanced_train_set <- balanced_train_set[sample(1:nrow(balanced_train_set)),]
table(balanced_train_set$salary)
write.csv(train_set, "data/train_data_full.csv")
write.csv(balanced_train_set, "data/train_data_balanced.csv")
write.csv(val_set, "data/validation_set.csv")
write.csv(val_set[salary==" 50000+."], "data/validation_set_positive.csv")

## Apply same transformations to test data and save
test$num_persons_worked_for_employer <- factor(test$num_persons_worked_for_employer, 
                                                levels=unique(train$num_persons_worked_for_employer))
test$own_business_or_self_employed <- factor(test$own_business_or_self_employed, 
                                              levels=unique(train$own_business_or_self_employed))
test$veterans_benefits <- factor(test$veterans_benefits, levels=unique(train$veterans_benefits))
test$year <- factor(test$year, levels=unique(train$year))
degrees <- c(" Masters degree(MA MS MEng MEd MSW MBA)", " Prof school degree (MD DDS DVM LLB JD)", " High school graduate",
             " Doctorate degree(PhD EdD)", " Bachelors degree(BA AB BS)", " Associates degree-academic program")
test[!(education %in% degrees)]$education <- "School"
test[education %in% c(" Prof school degree (MD DDS DVM LLB JD)"," Doctorate degree(PhD EdD)")]$education <- "Doctorate"
test[education %in% c(" Bachelors degree(BA AB BS)", " Associates degree-academic program")]$education <- "Bachelors"

test[grepl("householder", train$state_of_previous_residence)]$state_of_previous_residence <- "Householder"
test[!(grepl("householder", train$state_of_previous_residence))]$state_of_previous_residence <- "Not Householder"
test <- test[wage_per_hour <= 5000]
test <- test[divdends_from_stocks <= 25000]
write.csv(test, "data/test_set.csv")


####--------------------------------------------------
####----------- Part 3: Random Forest Model ----------
####--------------------------------------------------
# DONE USING SCIKIT-LEARN IN PYTHON


####--------------------------------------------------
####----------- Part 4: Evaluating Feature Importance
####--------------------------------------------------
importance <- data.table(read.csv('results/feature_importance.csv'))
setnames(importance, 'X', 'feature')
importance <- importance[order(-Importance)]
# levels(importance$feature) <- rev(unique(importance$feature))
importance$feature <- factor(as.character(importance$feature), rev(as.character(unique(importance$feature))))
g <- ggplot(data=importance, aes(x=feature, y=Importance, fill=feature)) + geom_bar(stat='identity')
g <- g + coord_flip() + guides(fill=FALSE)
g


## Boxplots and summary statistics of important features
train$salary_high <- ifelse(train$salary==" 50000+.", 1, 0)

# Occupation code
occ_data <- subset(train, select=c(occupation_code, salary_high))
occ_data$count <- 1
occ_data <- occ_data[, count_high:=sum(salary_high), by=occupation_code]
occ_data <- occ_data[, count_total:=sum(count), by=occupation_code]
occ_data <- unique(subset(occ_data, select=-c(salary_high, count)))
occ_data$pct_high_salary <- occ_data$count_high / occ_data$count_total
occ_data <- occ_data[order(-pct_high_salary)]
occ_data <- occ_data[1:20]
occ_data$occupation_code <- factor(occ_data$occupation_code, levels=unique(occ_data$occupation_code))
g <- ggplot(data=occ_data, aes(x=occupation_code, y=pct_high_salary, fill=occupation_code)) + geom_bar(stat='identity')
g <- g + coord_flip() + guides(fill=FALSE)
g + scale_x_discrete(limits = rev(levels(occ_data$occupation_code)))

# Weeks worked in year
summary(train[salary==" 50000+."]$weeks_worked_in_year)
summary(train[salary!=" 50000+."]$weeks_worked_in_year)

# Age
ggplot(data=train, aes(x=salary, y=age)) + geom_boxplot(fill='lightblue')






