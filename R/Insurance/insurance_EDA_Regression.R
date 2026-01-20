# =============================================================================
# Insurance Dataset Analysis - EDA, Preprocessing & Regression Models
# Dataset: https://www.kaggle.com/datasets/mirichoi0218/insurance/data
# =============================================================================

# Set seed for reproducibility
set.seed(42)

# -----------------------------------------------------------------------------
# 1. Load Required Libraries
# -----------------------------------------------------------------------------
library(tidyverse)    # Data manipulation and visualization
library(caret)        # Machine learning framework
library(corrplot)     # Correlation plots
library(gridExtra)    # Arrange multiple plots
library(randomForest) # Random Forest model
library(glmnet)       # Ridge/Lasso regression
library(Metrics)      # Evaluation metrics
library(moments)      # For skewness calculation
library(ipred)        # Bagging regression
library(gbm)          # Gradient Boosting Machine

# -----------------------------------------------------------------------------
# 2. Load and Inspect Data
# -----------------------------------------------------------------------------
insurance <- read.csv("insurance.csv")

# Basic structure
cat("=== Dataset Structure ===\n")
str(insurance)

cat("\n=== Dataset Dimensions ===\n")
cat("Rows:", nrow(insurance), "| Columns:", ncol(insurance), "\n")

cat("\n=== First Few Rows ===\n")
print(head(insurance))

cat("\n=== Summary Statistics ===\n")
print(summary(insurance))

# Check for missing values
missing_values <- colSums(is.na(insurance))
cat("\n=== Missing Values ===\n")
print(missing_values)

# Check for duplicates
n_duplicates <- sum(duplicated(insurance))
cat("\n=== Duplicate Rows ===\n")
cat("Number of duplicates:", n_duplicates, "\n")

# -----------------------------------------------------------------------------
# 3. Exploratory Data Analysis (EDA)
# -----------------------------------------------------------------------------

# 3.1 Distribution of Target Variable (charges)
p1 <- ggplot(insurance, aes(x = charges)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white", alpha = 0.7) +
  labs(title = "Distribution of Insurance Charges",
       x = "Charges ($)", y = "Frequency") +
  theme_minimal()

p2 <- ggplot(insurance, aes(x = log(charges))) +
  geom_histogram(bins = 50, fill = "darkgreen", color = "white", alpha = 0.7) +
  labs(title = "Distribution of Log-Transformed Charges",
       x = "Log(Charges)", y = "Frequency") +
  theme_minimal()

grid.arrange(p1, p2, ncol = 2)

# 3.2 Categorical Variables Distribution
p3 <- ggplot(insurance, aes(x = sex, fill = sex)) +
  geom_bar(alpha = 0.7) +
  labs(title = "Gender Distribution", x = "Sex", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

p4 <- ggplot(insurance, aes(x = smoker, fill = smoker)) +
  geom_bar(alpha = 0.7) +
  labs(title = "Smoker Distribution", x = "Smoker", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

p5 <- ggplot(insurance, aes(x = region, fill = region)) +
  geom_bar(alpha = 0.7) +
  labs(title = "Region Distribution", x = "Region", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none", axis.text.x = element_text(angle = 45, hjust = 1))

p6 <- ggplot(insurance, aes(x = factor(children), fill = factor(children))) +
  geom_bar(alpha = 0.7) +
  labs(title = "Children Distribution", x = "Number of Children", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

grid.arrange(p3, p4, p5, p6, ncol = 2)

# 3.3 Numerical Variables Distribution
p7 <- ggplot(insurance, aes(x = age)) +
  geom_histogram(bins = 30, fill = "coral", color = "white", alpha = 0.7) +
  labs(title = "Age Distribution", x = "Age", y = "Frequency") +
  theme_minimal()

p8 <- ggplot(insurance, aes(x = bmi)) +
  geom_histogram(bins = 30, fill = "purple", color = "white", alpha = 0.7) +
  labs(title = "BMI Distribution", x = "BMI", y = "Frequency") +
  theme_minimal()

grid.arrange(p7, p8, ncol = 2)

# 3.4 Relationship between features and target
p9 <- ggplot(insurance, aes(x = smoker, y = charges, fill = smoker)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Charges by Smoker Status", x = "Smoker", y = "Charges ($)") +
  theme_minimal() +
  theme(legend.position = "none")

p10 <- ggplot(insurance, aes(x = region, y = charges, fill = region)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Charges by Region", x = "Region", y = "Charges ($)") +
  theme_minimal() +
  theme(legend.position = "none")

grid.arrange(p9, p10, ncol = 2)

# 3.5 Scatter plots
p11 <- ggplot(insurance, aes(x = age, y = charges, color = smoker)) +
  geom_point(alpha = 0.6) +
  labs(title = "Age vs Charges (by Smoker)", x = "Age", y = "Charges ($)") +
  theme_minimal()

p12 <- ggplot(insurance, aes(x = bmi, y = charges, color = smoker)) +
  geom_point(alpha = 0.6) +
  labs(title = "BMI vs Charges (by Smoker)", x = "BMI", y = "Charges ($)") +
  theme_minimal()

grid.arrange(p11, p12, ncol = 2)

# 3.6 Correlation Matrix (numerical variables)
numerical_vars <- insurance %>% select(age, bmi, children, charges)
cor_matrix <- cor(numerical_vars)
cat("\n=== Correlation Matrix ===\n")
print(round(cor_matrix, 3))

corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black", tl.srt = 45,
         title = "Correlation Matrix", mar = c(0, 0, 1, 0))

# -----------------------------------------------------------------------------
# 4. DYNAMIC Insights Extraction from EDA
# -----------------------------------------------------------------------------

# Function to generate insights based on actual data
generate_eda_insights <- function(data) {
  insights <- list()

  # 1. Target variable skewness
  charges_skewness <- skewness(data$charges)
  log_charges_skewness <- skewness(log(data$charges))

  if (charges_skewness > 1) {
    skew_desc <- "highly right-skewed"
  } else if (charges_skewness > 0.5) {
    skew_desc <- "moderately right-skewed"
  } else if (charges_skewness < -0.5) {
    skew_desc <- "left-skewed"
  } else {
    skew_desc <- "approximately symmetric"
  }

  insights$target_skewness <- list(
    raw_skewness = round(charges_skewness, 3),
    log_skewness = round(log_charges_skewness, 3),
    description = skew_desc,
    recommend_log = abs(log_charges_skewness) < abs(charges_skewness)
  )

  # 2. Missing values analysis
  insights$missing_values <- list(
    total_missing = sum(is.na(data)),
    has_missing = any(is.na(data))
  )

  # 3. Duplicates analysis
  insights$duplicates <- list(
    count = sum(duplicated(data)),
    percentage = round(sum(duplicated(data)) / nrow(data) * 100, 2)
  )

  # 4. Smoker impact analysis
  smoker_charges <- data %>% filter(smoker == "yes") %>% pull(charges)
  nonsmoker_charges <- data %>% filter(smoker == "no") %>% pull(charges)

  smoker_mean <- mean(smoker_charges)
  nonsmoker_mean <- mean(nonsmoker_charges)
  smoker_ratio <- smoker_mean / nonsmoker_mean

  # Statistical test for smoker effect
  smoker_ttest <- t.test(smoker_charges, nonsmoker_charges)

  insights$smoker_effect <- list(
    smoker_mean = round(smoker_mean, 2),
    nonsmoker_mean = round(nonsmoker_mean, 2),
    ratio = round(smoker_ratio, 2),
    p_value = smoker_ttest$p.value,
    significant = smoker_ttest$p.value < 0.05
  )

  # 5. Correlation analysis with target
  cor_with_charges <- cor(data %>% select(age, bmi, children), data$charges)

  # Find strongest numerical predictor
  strongest_cor_idx <- which.max(abs(cor_with_charges))
  strongest_predictor <- rownames(cor_with_charges)[strongest_cor_idx]
  strongest_cor_value <- cor_with_charges[strongest_cor_idx]

  insights$correlations <- list(
    age = round(cor_with_charges["age", ], 3),
    bmi = round(cor_with_charges["bmi", ], 3),
    children = round(cor_with_charges["children", ], 3),
    strongest_numerical = strongest_predictor,
    strongest_cor = round(strongest_cor_value, 3)
  )

  # 6. Age analysis
  insights$age <- list(
    min = min(data$age),
    max = max(data$age),
    mean = round(mean(data$age), 1),
    median = median(data$age)
  )

  # 7. BMI analysis
  obese_count <- sum(data$bmi >= 30)
  obese_pct <- round(obese_count / nrow(data) * 100, 1)

  insights$bmi <- list(
    min = round(min(data$bmi), 1),
    max = round(max(data$bmi), 1),
    mean = round(mean(data$bmi), 1),
    obese_count = obese_count,
    obese_percentage = obese_pct
  )

  # 8. Region analysis
  region_means <- data %>%
    group_by(region) %>%
    summarise(mean_charges = mean(charges), .groups = "drop") %>%
    arrange(desc(mean_charges))

  insights$region <- list(
    highest_cost_region = region_means$region[1],
    highest_mean = round(region_means$mean_charges[1], 2),
    lowest_cost_region = region_means$region[nrow(region_means)],
    lowest_mean = round(region_means$mean_charges[nrow(region_means)], 2),
    region_diff_pct = round((region_means$mean_charges[1] / region_means$mean_charges[nrow(region_means)] - 1) * 100, 1)
  )

  # 9. Children analysis
  children_mode <- as.numeric(names(which.max(table(data$children))))
  children_cor <- cor(data$children, data$charges)

  insights$children <- list(
    mode = children_mode,
    max = max(data$children),
    correlation_with_charges = round(children_cor, 3),
    weak_predictor = abs(children_cor) < 0.1
  )

  # 10. BMI-Smoker interaction
  smoker_obese <- data %>% filter(smoker == "yes", bmi >= 30) %>% pull(charges)
  smoker_nonobese <- data %>% filter(smoker == "yes", bmi < 30) %>% pull(charges)
  nonsmoker_obese <- data %>% filter(smoker == "no", bmi >= 30) %>% pull(charges)
  nonsmoker_nonobese <- data %>% filter(smoker == "no", bmi < 30) %>% pull(charges)

  insights$bmi_smoker_interaction <- list(
    smoker_obese_mean = round(mean(smoker_obese), 2),
    smoker_nonobese_mean = round(mean(smoker_nonobese), 2),
    nonsmoker_obese_mean = round(mean(nonsmoker_obese), 2),
    nonsmoker_nonobese_mean = round(mean(nonsmoker_nonobese), 2),
    interaction_effect = round(mean(smoker_obese) - mean(smoker_nonobese), 2) >
                         round(mean(nonsmoker_obese) - mean(nonsmoker_nonobese), 2)
  )

  return(insights)
}

# Generate insights
eda_insights <- generate_eda_insights(insurance)

# Print dynamic insights
cat("\n")
cat("=============================================================================\n")
cat("                    KEY INSIGHTS FROM EDA (Data-Driven)                      \n")
cat("=============================================================================\n\n")

cat("1. TARGET VARIABLE (charges):\n")
cat("   - Skewness:", eda_insights$target_skewness$raw_skewness,
    "(", eda_insights$target_skewness$description, ")\n")
cat("   - Log-transformed skewness:", eda_insights$target_skewness$log_skewness, "\n")
if (eda_insights$target_skewness$recommend_log) {
  cat("   - RECOMMENDATION: Log transformation improves normality\n")
}

cat("\n2. DATA QUALITY:\n")
cat("   - Missing values:", eda_insights$missing_values$total_missing, "\n")
cat("   - Duplicate rows:", eda_insights$duplicates$count,
    "(", eda_insights$duplicates$percentage, "%)\n")

cat("\n3. SMOKER STATUS (Strongest Categorical Predictor):\n")
cat("   - Smoker mean charges: $", eda_insights$smoker_effect$smoker_mean, "\n", sep = "")
cat("   - Non-smoker mean charges: $", eda_insights$smoker_effect$nonsmoker_mean, "\n", sep = "")
cat("   - Smokers pay", eda_insights$smoker_effect$ratio, "x more on average\n")
cat("   - Statistical significance: p-value =",
    format(eda_insights$smoker_effect$p_value, scientific = TRUE), "\n")

cat("\n4. CORRELATIONS WITH CHARGES:\n")
cat("   - Age:", eda_insights$correlations$age, "\n")
cat("   - BMI:", eda_insights$correlations$bmi, "\n")
cat("   - Children:", eda_insights$correlations$children, "\n")
cat("   - Strongest numerical predictor:", eda_insights$correlations$strongest_numerical,
    "(r =", eda_insights$correlations$strongest_cor, ")\n")

cat("\n5. AGE:\n")
cat("   - Range:", eda_insights$age$min, "-", eda_insights$age$max, "years\n")
cat("   - Mean:", eda_insights$age$mean, "| Median:", eda_insights$age$median, "\n")

cat("\n6. BMI:\n")
cat("   - Range:", eda_insights$bmi$min, "-", eda_insights$bmi$max, "\n")
cat("   - Mean:", eda_insights$bmi$mean, "\n")
cat("   - Obese (BMI >= 30):", eda_insights$bmi$obese_count,
    "(", eda_insights$bmi$obese_percentage, "% of data)\n")

cat("\n7. REGION:\n")
cat("   - Highest avg charges:", eda_insights$region$highest_cost_region,
    "($", eda_insights$region$highest_mean, ")\n", sep = "")
cat("   - Lowest avg charges:", eda_insights$region$lowest_cost_region,
    "($", eda_insights$region$lowest_mean, ")\n", sep = "")
cat("   - Difference:", eda_insights$region$region_diff_pct, "%\n")

cat("\n8. CHILDREN:\n")
cat("   - Most common:", eda_insights$children$mode, "children\n")
cat("   - Correlation with charges:", eda_insights$children$correlation_with_charges)
if (eda_insights$children$weak_predictor) {
  cat(" (weak predictor)\n")
} else {
  cat("\n")
}

cat("\n9. BMI-SMOKER INTERACTION:\n")
cat("   - Smoker + Obese mean: $", eda_insights$bmi_smoker_interaction$smoker_obese_mean, "\n", sep = "")
cat("   - Smoker + Non-obese mean: $", eda_insights$bmi_smoker_interaction$smoker_nonobese_mean, "\n", sep = "")
cat("   - Non-smoker + Obese mean: $", eda_insights$bmi_smoker_interaction$nonsmoker_obese_mean, "\n", sep = "")
cat("   - Non-smoker + Non-obese mean: $", eda_insights$bmi_smoker_interaction$nonsmoker_nonobese_mean, "\n", sep = "")
if (eda_insights$bmi_smoker_interaction$interaction_effect) {
  cat("   - INTERACTION DETECTED: BMI effect is stronger for smokers\n")
}

cat("\n=============================================================================\n\n")

# -----------------------------------------------------------------------------
# 5. Data Preprocessing
# -----------------------------------------------------------------------------

df <- insurance

# Encode categorical variables
df$sex <- ifelse(df$sex == "male", 1, 0)
df$smoker <- ifelse(df$smoker == "yes", 1, 0)

# One-hot encode region
df <- df %>%
  mutate(
    region_northwest = ifelse(region == "northwest", 1, 0),
    region_southeast = ifelse(region == "southeast", 1, 0),
    region_southwest = ifelse(region == "southwest", 1, 0)
  ) %>%
  select(-region)

# Feature Engineering based on EDA insights
# Create interaction term if interaction was detected
if (eda_insights$bmi_smoker_interaction$interaction_effect) {
  df$bmi_smoker <- df$bmi * df$smoker
  cat("Added bmi_smoker interaction feature based on EDA findings.\n")
}

# Create age groups
df$age_group <- cut(df$age, breaks = c(0, 30, 45, 60, Inf),
                    labels = c(1, 2, 3, 4), include.lowest = TRUE)
df$age_group <- as.numeric(df$age_group)

cat("\n=== Preprocessed Data Structure ===\n")
str(df)

# -----------------------------------------------------------------------------
# 6. Split Data: Train (60%), Validation (20%), Test (20%)
# -----------------------------------------------------------------------------

n <- nrow(df)
train_idx <- sample(1:n, size = 0.6 * n)
remaining <- setdiff(1:n, train_idx)
val_idx <- sample(remaining, size = 0.5 * length(remaining))
test_idx <- setdiff(remaining, val_idx)

train_data <- df[train_idx, ]
val_data <- df[val_idx, ]
test_data <- df[test_idx, ]

cat("\n=== Data Split ===\n")
cat("Training set:", nrow(train_data), "samples (", round(nrow(train_data)/n*100, 1), "%)\n")
cat("Validation set:", nrow(val_data), "samples (", round(nrow(val_data)/n*100, 1), "%)\n")
cat("Test set:", nrow(test_data), "samples (", round(nrow(test_data)/n*100, 1), "%)\n")

X_train <- train_data %>% select(-charges)
y_train <- train_data$charges

X_val <- val_data %>% select(-charges)
y_val <- val_data$charges

X_test <- test_data %>% select(-charges)
y_test <- test_data$charges

# -----------------------------------------------------------------------------
# 7. Evaluation Metrics Function
# -----------------------------------------------------------------------------

evaluate_model <- function(actual, predicted, model_name) {
  rmse_val <- rmse(actual, predicted)
  mae_val <- mae(actual, predicted)
  r2_val <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  mape_val <- mean(abs((actual - predicted) / actual)) * 100

  cat("\n=== ", model_name, " Performance ===\n")
  cat("RMSE:", round(rmse_val, 2), "\n")
  cat("MAE:", round(mae_val, 2), "\n")
  cat("R-squared:", round(r2_val, 4), "\n")
  cat("MAPE:", round(mape_val, 2), "%\n")

  return(data.frame(
    Model = model_name,
    RMSE = rmse_val,
    MAE = mae_val,
    R_squared = r2_val,
    MAPE = mape_val
  ))
}

# -----------------------------------------------------------------------------
# 8. Model 1: Random Forest Regression (Ensemble - Bagging with feature sampling)
# -----------------------------------------------------------------------------
cat("\n")
cat("=============================================================================\n")
cat("            MODEL 1: RANDOM FOREST (Bagging + Feature Sampling)              \n")
cat("=============================================================================\n")

set.seed(42)

rf_grid <- expand.grid(
  ntree = c(100, 200, 500),
  mtry = c(2, 3, 4, 5)
)

best_rf_rmse <- Inf
best_rf_model <- NULL
best_rf_params <- NULL

cat("\nTuning Random Forest hyperparameters...\n")

for (i in 1:nrow(rf_grid)) {
  set.seed(42)
  rf_temp <- randomForest(
    x = X_train,
    y = y_train,
    ntree = rf_grid$ntree[i],
    mtry = rf_grid$mtry[i],
    importance = TRUE
  )

  pred_val <- predict(rf_temp, X_val)
  rmse_val <- rmse(y_val, pred_val)

  if (rmse_val < best_rf_rmse) {
    best_rf_rmse <- rmse_val
    best_rf_model <- rf_temp
    best_rf_params <- rf_grid[i, ]
  }
}

cat("\nBest Random Forest Parameters:\n")
cat("  ntree:", best_rf_params$ntree, "\n")
cat("  mtry:", best_rf_params$mtry, "\n")
cat("  Validation RMSE:", round(best_rf_rmse, 2), "\n")

# Feature Importance (dynamic)
importance_df <- data.frame(
  Feature = rownames(importance(best_rf_model)),
  Importance = importance(best_rf_model)[, "%IncMSE"]
) %>%
  arrange(desc(Importance))

cat("\n=== Feature Importance (Random Forest) ===\n")
print(importance_df)

# Identify top predictors dynamically
top_predictors_rf <- head(importance_df$Feature, 3)
cat("\nTop 3 predictors:", paste(top_predictors_rf, collapse = ", "), "\n")

ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
  coord_flip() +
  labs(title = "Random Forest - Feature Importance",
       x = "Feature", y = "% Increase in MSE") +
  theme_minimal()

rf_val_pred <- predict(best_rf_model, X_val)
rf_val_metrics <- evaluate_model(y_val, rf_val_pred, "Random Forest (Validation)")

# -----------------------------------------------------------------------------
# 9. Model 2: Ridge Regression (Linear - Regularized)
# -----------------------------------------------------------------------------
cat("\n")
cat("=============================================================================\n")
cat("                   MODEL 2: RIDGE REGRESSION (Linear)                        \n")
cat("=============================================================================\n")

X_train_matrix <- as.matrix(X_train)
X_val_matrix <- as.matrix(X_val)
X_test_matrix <- as.matrix(X_test)

set.seed(42)
ridge_cv <- cv.glmnet(
  x = X_train_matrix,
  y = y_train,
  alpha = 0,
  nfolds = 10
)

cat("\nOptimal Lambda (min):", round(ridge_cv$lambda.min, 4), "\n")
cat("Optimal Lambda (1se):", round(ridge_cv$lambda.1se, 4), "\n")

plot(ridge_cv, main = "Ridge Regression - Cross-Validation")

ridge_model <- glmnet(
  x = X_train_matrix,
  y = y_train,
  alpha = 0,
  lambda = ridge_cv$lambda.min
)

# Extract and analyze coefficients dynamically
ridge_coef <- coef(ridge_model)
coef_df <- data.frame(
  Feature = rownames(ridge_coef),
  Coefficient = as.vector(ridge_coef)
) %>%
  filter(Feature != "(Intercept)") %>%
  arrange(desc(abs(Coefficient)))

cat("\n=== Ridge Regression Coefficients (sorted by magnitude) ===\n")
print(coef_df)

# Identify most influential features from Ridge
top_ridge_features <- head(coef_df$Feature, 3)
cat("\nTop 3 features by coefficient magnitude:", paste(top_ridge_features, collapse = ", "), "\n")

ridge_val_pred <- predict(ridge_model, X_val_matrix, s = ridge_cv$lambda.min)
ridge_val_metrics <- evaluate_model(y_val, ridge_val_pred, "Ridge Regression (Validation)")

# -----------------------------------------------------------------------------
# 10. Model 3: Bagging Regression (Pure Bagging)
# -----------------------------------------------------------------------------
cat("\n")
cat("=============================================================================\n")
cat("                    MODEL 3: BAGGING REGRESSION (ipred)                      \n")
cat("=============================================================================\n")

set.seed(42)

# Tune number of bagging iterations using validation set
bag_nbagg_options <- c(25, 50, 100, 150)
best_bag_rmse <- Inf
best_bag_model <- NULL
best_bag_nbagg <- NULL

cat("\nTuning Bagging hyperparameters...\n")

for (nbagg in bag_nbagg_options) {
  set.seed(42)
  bag_temp <- bagging(
    charges ~ .,
    data = train_data,
    nbagg = nbagg,
    coob = TRUE  # Out-of-bag error estimation
  )

  pred_val <- predict(bag_temp, val_data)
  rmse_val <- rmse(y_val, pred_val)

  cat("  nbagg =", nbagg, "-> Validation RMSE:", round(rmse_val, 2), "\n")

  if (rmse_val < best_bag_rmse) {
    best_bag_rmse <- rmse_val
    best_bag_model <- bag_temp
    best_bag_nbagg <- nbagg
  }
}

cat("\nBest Bagging Parameters:\n")
cat("  nbagg:", best_bag_nbagg, "\n")
cat("  Validation RMSE:", round(best_bag_rmse, 2), "\n")

bag_val_pred <- predict(best_bag_model, val_data)
bag_val_metrics <- evaluate_model(y_val, bag_val_pred, "Bagging (Validation)")

# -----------------------------------------------------------------------------
# 11. Model 4: Gradient Boosting Machine (GBM - Boosting)
# -----------------------------------------------------------------------------
cat("\n")
cat("=============================================================================\n")
cat("                MODEL 4: GRADIENT BOOSTING MACHINE (Boosting)                \n")
cat("=============================================================================\n")

set.seed(42)

# Define hyperparameter grid for GBM
gbm_grid <- expand.grid(
  n.trees = c(100, 200, 500),
  interaction.depth = c(3, 5, 7),
  shrinkage = c(0.01, 0.1)
)

best_gbm_rmse <- Inf
best_gbm_model <- NULL
best_gbm_params <- NULL
best_gbm_ntrees <- NULL

cat("\nTuning GBM hyperparameters (this may take a moment)...\n")

for (i in 1:nrow(gbm_grid)) {
  set.seed(42)
  gbm_temp <- gbm(
    charges ~ .,
    data = train_data,
    distribution = "gaussian",
    n.trees = gbm_grid$n.trees[i],
    interaction.depth = gbm_grid$interaction.depth[i],
    shrinkage = gbm_grid$shrinkage[i],
    n.minobsinnode = 10,
    cv.folds = 5,
    verbose = FALSE
  )

  # Find optimal number of trees using CV
  best_iter <- gbm.perf(gbm_temp, method = "cv", plot.it = FALSE)

  pred_val <- predict(gbm_temp, val_data, n.trees = best_iter)
  rmse_val <- rmse(y_val, pred_val)

  if (rmse_val < best_gbm_rmse) {
    best_gbm_rmse <- rmse_val
    best_gbm_model <- gbm_temp
    best_gbm_params <- gbm_grid[i, ]
    best_gbm_ntrees <- best_iter
  }
}

cat("\nBest GBM Parameters:\n")
cat("  n.trees (optimal via CV):", best_gbm_ntrees, "\n")
cat("  interaction.depth:", best_gbm_params$interaction.depth, "\n")
cat("  shrinkage (learning rate):", best_gbm_params$shrinkage, "\n")
cat("  Validation RMSE:", round(best_gbm_rmse, 2), "\n")

# Feature Importance from GBM
gbm_importance <- summary(best_gbm_model, n.trees = best_gbm_ntrees, plotit = FALSE)
cat("\n=== Feature Importance (GBM) ===\n")
print(gbm_importance)

top_predictors_gbm <- head(gbm_importance$var, 3)
cat("\nTop 3 predictors:", paste(top_predictors_gbm, collapse = ", "), "\n")

# Plot GBM feature importance
ggplot(gbm_importance, aes(x = reorder(var, rel.inf), y = rel.inf)) +
  geom_bar(stat = "identity", fill = "darkgreen", alpha = 0.7) +
  coord_flip() +
  labs(title = "Gradient Boosting - Feature Importance",
       x = "Feature", y = "Relative Influence (%)") +
  theme_minimal()

gbm_val_pred <- predict(best_gbm_model, val_data, n.trees = best_gbm_ntrees)
gbm_val_metrics <- evaluate_model(y_val, gbm_val_pred, "GBM (Validation)")

# -----------------------------------------------------------------------------
# 12. Model Comparison on Validation Set
# -----------------------------------------------------------------------------
cat("\n")
cat("=============================================================================\n")
cat("                    MODEL COMPARISON (VALIDATION SET)                        \n")
cat("=============================================================================\n")

comparison_val <- rbind(rf_val_metrics, ridge_val_metrics, bag_val_metrics, gbm_val_metrics)
print(comparison_val)

comparison_long <- comparison_val %>%
  pivot_longer(cols = c(RMSE, MAE, R_squared, MAPE),
               names_to = "Metric", values_to = "Value")

ggplot(comparison_long, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  facet_wrap(~Metric, scales = "free") +
  labs(title = "Model Comparison on Validation Set (4 Models)") +
  theme_minimal() +
  theme(axis.text.x = element_blank(),
        legend.position = "bottom")

# -----------------------------------------------------------------------------
# 13. Final Evaluation on Test Set
# -----------------------------------------------------------------------------
cat("\n")
cat("=============================================================================\n")
cat("                     FINAL EVALUATION (TEST SET)                             \n")
cat("=============================================================================\n")

# Random Forest
rf_test_pred <- predict(best_rf_model, X_test)
rf_test_metrics <- evaluate_model(y_test, rf_test_pred, "Random Forest (Test)")

# Ridge Regression
ridge_test_pred <- predict(ridge_model, X_test_matrix, s = ridge_cv$lambda.min)
ridge_test_metrics <- evaluate_model(y_test, ridge_test_pred, "Ridge Regression (Test)")

# Bagging
bag_test_pred <- predict(best_bag_model, test_data)
bag_test_metrics <- evaluate_model(y_test, bag_test_pred, "Bagging (Test)")

# GBM
gbm_test_pred <- predict(best_gbm_model, test_data, n.trees = best_gbm_ntrees)
gbm_test_metrics <- evaluate_model(y_test, gbm_test_pred, "GBM (Test)")

# -----------------------------------------------------------------------------
# 14. DYNAMIC Final Conclusion (4 Models)
# -----------------------------------------------------------------------------

comparison_test <- rbind(rf_test_metrics, ridge_test_metrics, bag_test_metrics, gbm_test_metrics)

cat("\n")
cat("=============================================================================\n")
cat("                        FINAL MODEL COMPARISON                               \n")
cat("=============================================================================\n")
print(comparison_test)

# Rank models by each metric
comparison_test$RMSE_rank <- rank(comparison_test$RMSE)
comparison_test$MAE_rank <- rank(comparison_test$MAE)
comparison_test$R2_rank <- rank(-comparison_test$R_squared)  # Higher is better
comparison_test$MAPE_rank <- rank(comparison_test$MAPE)
comparison_test$Total_score <- comparison_test$RMSE_rank + comparison_test$MAE_rank +
                               comparison_test$R2_rank + comparison_test$MAPE_rank

# Find the winner (lowest total score = best)
winner_idx <- which.min(comparison_test$Total_score)
winner <- comparison_test$Model[winner_idx]
winner_r2 <- comparison_test$R_squared[winner_idx]
winner_rmse <- comparison_test$RMSE[winner_idx]
winner_mae <- comparison_test$MAE[winner_idx]

# Find second best
comparison_test_sorted <- comparison_test %>% arrange(Total_score)
second_best <- comparison_test_sorted$Model[2]
second_r2 <- comparison_test_sorted$R_squared[2]

# Calculate metrics for analysis
r2_improvement <- round((winner_r2 - second_r2) * 100, 2)

# Identify model categories
ensemble_models <- c("Random Forest (Test)", "Bagging (Test)", "GBM (Test)")
linear_models <- c("Ridge Regression (Test)")
bagging_models <- c("Random Forest (Test)", "Bagging (Test)")
boosting_models <- c("GBM (Test)")

# Check if ensemble outperformed linear
ensemble_avg_r2 <- mean(comparison_test$R_squared[comparison_test$Model %in% ensemble_models])
linear_avg_r2 <- mean(comparison_test$R_squared[comparison_test$Model %in% linear_models])
ensemble_wins <- ensemble_avg_r2 > linear_avg_r2

# Check bagging vs boosting
bagging_avg_r2 <- mean(comparison_test$R_squared[comparison_test$Model %in% bagging_models])
boosting_r2 <- comparison_test$R_squared[comparison_test$Model %in% boosting_models]

cat("\n")
cat("=============================================================================\n")
cat("                         CONCLUSION (Data-Driven)                            \n")
cat("=============================================================================\n\n")

cat("MODEL RANKING (by combined metrics - lower is better):\n")
for (i in 1:nrow(comparison_test_sorted)) {
  cat("  ", i, ". ", comparison_test_sorted$Model[i],
      " (R2: ", round(comparison_test_sorted$R_squared[i], 4),
      ", RMSE: $", round(comparison_test_sorted$RMSE[i], 2), ")\n", sep = "")
}

cat("\n")
cat("BEST MODEL:", winner, "\n")
cat("  - R-squared:", round(winner_r2, 4), "\n")
cat("  - RMSE: $", round(winner_rmse, 2), "\n", sep = "")
cat("  - MAE: $", round(winner_mae, 2), "\n", sep = "")
cat("  - Outperforms", second_best, "by", abs(r2_improvement), "percentage points (R2)\n")

cat("\n")
cat("-----------------------------------------------------------------------------\n")
cat("                          KEY FINDINGS                                       \n")
cat("-----------------------------------------------------------------------------\n\n")

# Finding 1: Ensemble vs Linear
cat("1. ENSEMBLE vs LINEAR MODELS:\n")
if (ensemble_wins) {
  cat("   - Ensemble methods (avg R2:", round(ensemble_avg_r2, 4),
      ") OUTPERFORM linear (R2:", round(linear_avg_r2, 4), ")\n")
  cat("   - Suggests non-linear relationships in data\n\n")
} else {
  cat("   - Linear model (R2:", round(linear_avg_r2, 4),
      ") competitive with ensembles (avg R2:", round(ensemble_avg_r2, 4), ")\n")
  cat("   - Relationships may be largely linear\n\n")
}

# Finding 2: Bagging vs Boosting
cat("2. BAGGING vs BOOSTING:\n")
cat("   - Bagging methods avg R2:", round(bagging_avg_r2, 4), "\n")
cat("   - Boosting (GBM) R2:", round(boosting_r2, 4), "\n")
if (boosting_r2 > bagging_avg_r2) {
  cat("   - Boosting performs better - sequential error correction helps\n\n")
} else {
  cat("   - Bagging performs better - variance reduction more important here\n\n")
}

# Finding 3: Smoker effect from EDA
cat("3. SMOKER STATUS IMPACT:\n")
if (eda_insights$smoker_effect$significant) {
  cat("   - Statistically significant (p < 0.05)\n")
  cat("   - Smokers pay ", eda_insights$smoker_effect$ratio, "x more ($",
      eda_insights$smoker_effect$smoker_mean, " vs $",
      eda_insights$smoker_effect$nonsmoker_mean, ")\n\n", sep = "")
} else {
  cat("   - Not statistically significant\n\n")
}

# Finding 4: Top predictors comparison
cat("4. TOP PREDICTORS ACROSS MODELS:\n")
cat("   - Random Forest:", paste(top_predictors_rf, collapse = ", "), "\n")
cat("   - GBM:", paste(top_predictors_gbm, collapse = ", "), "\n")
cat("   - Ridge:", paste(top_ridge_features, collapse = ", "), "\n")

# Check if models agree on top predictor
all_top1 <- c(top_predictors_rf[1], top_predictors_gbm[1], top_ridge_features[1])
if (length(unique(all_top1)) == 1) {
  cat("   - All models agree:", unique(all_top1), "is most important\n\n")
} else {
  cat("   - Models show different importance rankings\n\n")
}

# Finding 5: BMI-Smoker Interaction
cat("5. BMI-SMOKER INTERACTION:\n")
if (eda_insights$bmi_smoker_interaction$interaction_effect) {
  cat("   - Interaction detected and used as feature\n")
  cat("   - Obese smokers: $", eda_insights$bmi_smoker_interaction$smoker_obese_mean, "\n", sep = "")
  cat("   - Non-obese smokers: $", eda_insights$bmi_smoker_interaction$smoker_nonobese_mean, "\n\n", sep = "")
} else {
  cat("   - No significant interaction detected\n\n")
}

# Finding 6: Data quality
cat("6. DATA QUALITY:\n")
if (!eda_insights$missing_values$has_missing) {
  cat("   - No missing values\n")
}
if (eda_insights$duplicates$count > 0) {
  cat("   - ", eda_insights$duplicates$count, " duplicate rows found\n", sep = "")
} else {
  cat("   - No duplicates\n")
}

cat("\n=============================================================================\n")

# -----------------------------------------------------------------------------
# 15. Visualization of All Models
# -----------------------------------------------------------------------------

# Prepare predictions from all models for comparison
all_predictions <- data.frame(
  Actual = y_test,
  RandomForest = rf_test_pred,
  Ridge = as.vector(ridge_test_pred),
  Bagging = bag_test_pred,
  GBM = gbm_test_pred
)

# Reshape for plotting
all_pred_long <- all_predictions %>%
  pivot_longer(cols = -Actual, names_to = "Model", values_to = "Predicted")

# Actual vs Predicted for all models
ggplot(all_pred_long, aes(x = Actual, y = Predicted, color = Model)) +
  geom_point(alpha = 0.4) +
  geom_abline(intercept = 0, slope = 1, color = "black", linetype = "dashed", linewidth = 1) +
  facet_wrap(~Model, ncol = 2) +
  labs(title = "Actual vs Predicted Charges - All Models",
       x = "Actual Charges ($)", y = "Predicted Charges ($)") +
  theme_minimal() +
  theme(legend.position = "none")

# Best model visualization
if (grepl("Random Forest", winner)) {
  best_pred <- rf_test_pred
} else if (grepl("Ridge", winner)) {
  best_pred <- as.vector(ridge_test_pred)
} else if (grepl("Bagging", winner)) {
  best_pred <- bag_test_pred
} else {
  best_pred <- gbm_test_pred
}

results_df <- data.frame(Actual = y_test, Predicted = best_pred)
results_df$Residual <- results_df$Actual - results_df$Predicted

ggplot(results_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "steelblue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) +
  labs(title = paste("Actual vs Predicted Charges -", winner),
       subtitle = paste("R-squared:", round(winner_r2, 4), "| RMSE: $", round(winner_rmse, 2)),
       x = "Actual Charges ($)", y = "Predicted Charges ($)") +
  theme_minimal()

ggplot(results_df, aes(x = Predicted, y = Residual)) +
  geom_point(alpha = 0.5, color = "coral") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = paste("Residual Plot -", winner),
       x = "Predicted Charges ($)", y = "Residuals") +
  theme_minimal()

# Final comparison bar chart
comparison_test_clean <- comparison_test %>%
  select(Model, RMSE, MAE, R_squared, MAPE)

comparison_final_long <- comparison_test_clean %>%
  pivot_longer(cols = c(RMSE, MAE, R_squared, MAPE),
               names_to = "Metric", values_to = "Value")

ggplot(comparison_final_long, aes(x = reorder(Model, -Value), y = Value, fill = Model)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  facet_wrap(~Metric, scales = "free") +
  labs(title = "Final Model Comparison (Test Set)",
       x = "", y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")
