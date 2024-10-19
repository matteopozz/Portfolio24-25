# libraries
install.packages("dplyr")
install.packages("ggplot2", dependencies = TRUE)
install.packages("corrplot")
install.packages("RColorBrewer")
install.packages("GGally")
install.packages("moments")
install.packages("e1071")
library(e1071)
library(readxl)
library(dplyr)
library(ggplot2)
library(GGally)
library(RColorBrewer)
library(moments)
library(corrplot)
library(mgcv)

# Set working directory to the current script's location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Import Data
data <- read.csv("Pozzuto_Matteo_DA301_Assignment/reviews_clean.csv")

# Have imported the cleaned file for ease, but have included example code for cleaning.

# Dropping unnecessary columns
data <- subset(data, select = -c(review, summary))

# Renaming columns using dplyr
## data <- data %>% rename(
##  remuneration = remuneration (k£),
##  spending_score = spending_score (1-100)

# NA and duplicate checks
any(is.na(data))

# Count of Na per column
colSums(is.na(data))

# Check duplicated
any(duplicated(data))

# Count duplicated
sum(duplicated(data))

# Checking duplicated rows
duplicate_rows <- data[duplicated(data), ]

# Isolating and checking
head(duplicate_rows)

## Not real duplicates as different reviews/summaries.
## Not removing them as they don't seem to be a problem, and the dataset is already small.

# View data
head(data)

# Create a summary of the new data frame
summary(data)

# Viewing datatypes
column_data_types <- sapply(data, class)
print(column_data_types)

# Distribution of Loyalty Points
ggplot(data, aes(x = loyalty_points)) +
  geom_histogram(binwidth = 5, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Distribution of Loyalty Points", x = "Loyalty Points", y = "Count")

# Scatterplot of Loyalty Points vs Remuneration
ggplot(data, aes(x = remuneration, y = loyalty_points)) +
  geom_point(alpha = 0.5) +
  labs(title = "Loyalty Points vs Remuneration", x = "Remuneration", y = "Loyalty Points")

# Scatterplot of Loyalty Points vs Spending Score
ggplot(data, aes(x = spending_score, y = loyalty_points)) +
  geom_point(alpha = 0.5) +
  labs(title = "Loyalty Points vs Spending Score", x = "Spending Score", y = "Loyalty Points")

# Scatterplot of Loyalty Points vs Age
ggplot(data, aes(x = age, y = loyalty_points)) +
  geom_point(alpha = 0.5) +
  labs(title = "Loyalty Points vs Age", x = "Age", y = "Loyalty Points")

# Boxplot of Loyalty Points by Education
ggplot(data, aes(x = as.factor(education), y = loyalty_points)) +
  geom_boxplot() +
  labs(title = "Boxplot of Loyalty Points by Education", x = "Education", y = "Loyalty Points")

# Doughnut chart of distribution of education
# Calculate value counts in new table
education_counts <- as.data.frame(table(data$education))
colnames(education_counts) <- c("education", "count")

# Calculate percentages
education_counts$fraction <- education_counts$count / sum(education_counts$count)
education_counts$ymax <- cumsum(education_counts$fraction)
education_counts$ymin <- c(0, head(education_counts$ymax, n=-1))

# Create the donut chart
ggplot(education_counts, aes(ymax=ymax, ymin=ymin, xmax=4, xmin=3, fill=education)) +
  geom_rect() +
  coord_polar(theta="y") +
  xlim(c(2, 4)) +
  theme_void() +
  theme(legend.position = "right") +
  labs(fill="Education") +
  scale_fill_brewer(palette="Set1")

# Drop unnecessary columns for correlation analysis
numeric_data <- data %>%
  select_if(is.numeric)

# Calculate the correlation matrix and plot
correlation_matrix <- cor(numeric_data, use = "complete.obs")
corrplot(correlation_matrix, method = "circle")

# Creating pairplot to quickly visualise all data
ggpairs(data[, c('loyalty_points', 'remuneration', 'spending_score', 'product', 'age')],
        aes(alpha = 0.3),
        upper = list(continuous = wrap("cor", size = 3)),
        lower = list(continuous = "smooth"),
        diag = list(continuous = "barDiag"))

# Shapiro-Wilk test for normality of loyalty points
shapiro.test(data$loyalty_points)

# Data not normally distributed based on test. Checking skewness and kurtosis.

# Skewness and Kurtosis
skewness_value <- skewness(data$loyalty_points)
kurtosis_value <- kurtosis(data$loyalty_points)
print(skewness_value)
print(kurtosis_value)

# loyalty_points is moderately skewed to the right, average person has quite low loyalty points.
# Kurtosis is higher than normal, with heavy tails.

# Calculate Range
range_loyalty_points <- range(data$loyalty_points)

# Calculate Difference between highest and lowest values
difference_high_low <- diff(range_loyalty_points)

# Calculate Interquartile Range (IQR)
iqr_loyalty_points <- IQR(data$loyalty_points)

# Calculate Variance
variance_loyalty_points <- var(data$loyalty_points)

# Calculate Standard Deviation
std_deviation_loyalty_points <- sd(data$loyalty_points)

# Display results
list(
  Range = range_loyalty_points,
  Difference = difference_high_low,
  IQR = iqr_loyalty_points,
  Variance = variance_loyalty_points,
  Standard_Deviation = std_deviation_loyalty_points
)

# Extreme values to deal with, but data still centralised enough around the mean.

# More Measures of Shape
scores <- data$loyalty_points

# Calculate mean, median, and mode
mean_score <- mean(scores)
median_score <- median(scores)
mode_score <- as.numeric(names(sort(table(scores), decreasing = TRUE)[1]))

# Print the results
cat("Mean:", mean_score, "\n")
cat("Median:", median_score, "\n")
cat("Mode:", mode_score, "\n")

# Log transforming loyalty points for MLR to attempt to correct positive skew
data$log_loyalty_points <- log(data$loyalty_points + 1)

# Create the multiple linear regression model
model <- lm(log_loyalty_points ~ remuneration + spending_score + product + age, data = data)

# Summarize the model
summary(model)

# Removing product as statistically insignificant.
model2 <- lm(log_loyalty_points ~ remuneration + spending_score + age, data = data)
summary(model2)

# MRS is 81%, so looking pretty good!

# Plot actual vs. predicted values with a different smoothing method if necessary
ggplot(data, aes(x = log_loyalty_points, y = predict(model2, data))) +
  geom_point() +
  stat_smooth(method = "loess") +  # You can change 'loess' to 'lm' or other methods
  labs(x = 'Actual Loyalty Points', y = 'Predicted Loyalty Points') +
  ggtitle('Actual vs. Predicted Loyalty Points')

# Putting myself into model to test predictive power
new_data <- data.frame(remuneration = c(29.5), age = c(30), spending_score = c(55))
predictions <- predict(model2, new_data)
print(predictions)

# Model predicts my loyalty points, adjusted for LOG, would be 6.6, MLR has strong predictive power.