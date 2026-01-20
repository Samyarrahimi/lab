# =============================================================================
# Pima Indians Diabetes Classification - EDA, Preprocessing & ML Models
# Dataset: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
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
library(glmnet)       # Logistic Regression with regularization
library(e1071)        # SVM
library(xgboost)      # XGBoost (Boosting)
library(pROC)         # ROC curves and AUC
library(PRROC)        # Precision-Recall curves
library(moments)      # Skewness calculation
library(class)        # KNN

# -----------------------------------------------------------------------------
# 2. Load and Inspect Data
# -----------------------------------------------------------------------------
diabetes <- read.csv("diabetes.csv")

# Basic structure
cat("=== Dataset Structure ===\n")
str(diabetes)

cat("\n=== Dataset Dimensions ===\n")
cat("Rows:", nrow(diabetes), "| Columns:", ncol(diabetes), "\n")

cat("\n=== First Few Rows ===\n")
print(head(diabetes))

cat("\n=== Summary Statistics ===\n")
print(summary(diabetes))

# Check for missing values (explicit NA)
missing_values <- colSums(is.na(diabetes))
cat("\n=== Missing Values (NA) ===\n")
print(missing_values)

# Check for zeros in columns where zero is biologically impossible
cat("\n=== Zero Values Analysis (potential missing data) ===\n")
zero_cols <- c("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
for (col in zero_cols) {
  zero_count <- sum(diabetes[[col]] == 0)
  zero_pct <- round(zero_count / nrow(diabetes) * 100, 1)
  cat(col, ": ", zero_count, " zeros (", zero_pct, "%)\n", sep = "")
}

# Check for duplicates
n_duplicates <- sum(duplicated(diabetes))
cat("\n=== Duplicate Rows ===\n")
cat("Number of duplicates:", n_duplicates, "\n")

# Target variable distribution
cat("\n=== Target Variable Distribution ===\n")
print(table(diabetes$Outcome))
cat("Class imbalance ratio:",
    round(sum(diabetes$Outcome == 0) / sum(diabetes$Outcome == 1), 2), ": 1\n")

# -----------------------------------------------------------------------------
# 3. Exploratory Data Analysis (EDA)
# -----------------------------------------------------------------------------

# 3.1 Target Variable Distribution
p1 <- ggplot(diabetes, aes(x = factor(Outcome), fill = factor(Outcome))) +
  geom_bar(alpha = 0.7) +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5) +
  scale_fill_manual(values = c("steelblue", "coral"), labels = c("No Diabetes", "Diabetes")) +
  labs(title = "Distribution of Diabetes Outcome",
       x = "Outcome", y = "Count", fill = "Class") +
  theme_minimal()

print(p1)

# 3.2 Distribution of All Features
feature_cols <- setdiff(names(diabetes), "Outcome")

# Histograms for all features
plot_list <- list()
for (i in seq_along(feature_cols)) {
  col <- feature_cols[i]
  p <- ggplot(diabetes, aes_string(x = col)) +
    geom_histogram(bins = 30, fill = "steelblue", color = "white", alpha = 0.7) +
    labs(title = col, x = col, y = "Frequency") +
    theme_minimal() +
    theme(plot.title = element_text(size = 10))
  plot_list[[i]] <- p
}
do.call(grid.arrange, c(plot_list, ncol = 3))

# 3.3 Features by Outcome (Boxplots)
plot_list_box <- list()
for (i in seq_along(feature_cols)) {
  col <- feature_cols[i]
  p <- ggplot(diabetes, aes_string(x = "factor(Outcome)", y = col, fill = "factor(Outcome)")) +
    geom_boxplot(alpha = 0.7) +
    scale_fill_manual(values = c("steelblue", "coral")) +
    labs(title = col, x = "Outcome", y = col) +
    theme_minimal() +
    theme(legend.position = "none", plot.title = element_text(size = 10))
  plot_list_box[[i]] <- p
}
do.call(grid.arrange, c(plot_list_box, ncol = 3))

# 3.4 Linear Correlation Matrix
cat("\n=== Linear Correlation Matrix (Pearson) ===\n")
cor_matrix <- cor(diabetes)
print(round(cor_matrix, 3))

corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black", tl.srt = 45,
         title = "Linear Correlation Matrix (Pearson)", mar = c(0, 0, 1, 0),
         number.cex = 0.7)

# 3.5 Non-Linear Correlation Analysis (Spearman)
cat("\n=== Non-Linear Correlation Matrix (Spearman) ===\n")
cor_matrix_spearman <- cor(diabetes, method = "spearman")
print(round(cor_matrix_spearman, 3))

corrplot(cor_matrix_spearman, method = "color", type = "upper",
         addCoef.col = "black", tl.col = "black", tl.srt = 45,
         title = "Non-Linear Correlation Matrix (Spearman)", mar = c(0, 0, 1, 0),
         number.cex = 0.7)

# 3.6 Compare Pearson vs Spearman to detect non-linear relationships
cat("\n=== Correlation Comparison (Pearson vs Spearman) ===\n")
cat("Large differences suggest non-linear relationships:\n")
cor_diff <- abs(cor_matrix - cor_matrix_spearman)
# Get upper triangle pairs
for (i in 1:(ncol(cor_diff) - 1)) {
  for (j in (i + 1):ncol(cor_diff)) {
    if (cor_diff[i, j] > 0.05) {  # Threshold for notable difference
      cat("  ", rownames(cor_diff)[i], " - ", colnames(cor_diff)[j],
          ": Pearson=", round(cor_matrix[i, j], 3),
          ", Spearman=", round(cor_matrix_spearman[i, j], 3),
          ", Diff=", round(cor_diff[i, j], 3), "\n", sep = "")
    }
  }
}

# 3.7 Mutual Information (for non-linear dependency detection)
# Using discretized mutual information approximation
calculate_mutual_info <- function(x, y, bins = 10) {
  # Discretize continuous variables
  x_disc <- cut(x, breaks = bins, labels = FALSE)
  y_disc <- if (length(unique(y)) > bins) cut(y, breaks = bins, labels = FALSE) else y

  # Joint and marginal probabilities
  joint <- table(x_disc, y_disc) / length(x)
  px <- rowSums(joint)
  py <- colSums(joint)

  # Mutual information
  mi <- 0
  for (i in 1:nrow(joint)) {
    for (j in 1:ncol(joint)) {
      if (joint[i, j] > 0 && px[i] > 0 && py[j] > 0) {
        mi <- mi + joint[i, j] * log2(joint[i, j] / (px[i] * py[j]))
      }
    }
  }
  return(mi)
}

# Calculate MI between each feature and Outcome
cat("\n=== Mutual Information with Outcome (Non-Linear Dependency) ===\n")
mi_scores <- sapply(feature_cols, function(col) {
  calculate_mutual_info(diabetes[[col]], diabetes$Outcome)
})
mi_df <- data.frame(Feature = names(mi_scores), MI = mi_scores) %>%
  arrange(desc(MI))
print(mi_df)

# Plot MI scores
ggplot(mi_df, aes(x = reorder(Feature, MI), y = MI)) +
  geom_bar(stat = "identity", fill = "darkgreen", alpha = 0.7) +
  coord_flip() +
  labs(title = "Mutual Information with Outcome (Non-Linear Dependency)",
       x = "Feature", y = "Mutual Information (bits)") +
  theme_minimal()

# 3.8 Scatter plots with LOESS smoothing to visualize non-linear patterns
plot_list_scatter <- list()
for (i in seq_along(feature_cols)) {
  col <- feature_cols[i]
  p <- ggplot(diabetes, aes_string(x = col, y = "Outcome")) +
    geom_point(alpha = 0.3, color = "steelblue") +
    geom_smooth(method = "loess", color = "red", se = TRUE) +
    labs(title = paste(col, "vs Outcome (LOESS)"), x = col, y = "Outcome") +
    theme_minimal() +
    theme(plot.title = element_text(size = 9))
  plot_list_scatter[[i]] <- p
}
do.call(grid.arrange, c(plot_list_scatter, ncol = 3))

# -----------------------------------------------------------------------------
# 4. DYNAMIC Insights Extraction from EDA
# -----------------------------------------------------------------------------

generate_eda_insights <- function(data, feature_cols, mi_df, cor_matrix, cor_matrix_spearman) {
  insights <- list()

  # 1. Target class distribution
  class_counts <- table(data$Outcome)
  insights$class_balance <- list(
    negative = as.numeric(class_counts[1]),
    positive = as.numeric(class_counts[2]),
    imbalance_ratio = round(as.numeric(class_counts[1]) / as.numeric(class_counts[2]), 2),
    positive_rate = round(as.numeric(class_counts[2]) / nrow(data) * 100, 1)
  )

  # 2. Missing/Zero value analysis
  zero_cols <- c("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
  zero_analysis <- sapply(zero_cols, function(col) sum(data[[col]] == 0))
  insights$zero_values <- list(
    columns = zero_cols,
    counts = zero_analysis,
    worst_column = names(which.max(zero_analysis)),
    worst_count = max(zero_analysis),
    worst_pct = round(max(zero_analysis) / nrow(data) * 100, 1)
  )

  # 3. Skewness analysis
  skewness_vals <- sapply(feature_cols, function(col) skewness(data[[col]]))
  highly_skewed <- names(skewness_vals[abs(skewness_vals) > 1])
  insights$skewness <- list(
    values = round(skewness_vals, 3),
    highly_skewed = highly_skewed,
    most_skewed = names(which.max(abs(skewness_vals))),
    max_skewness = round(max(abs(skewness_vals)), 3)
  )

  # 4. Linear correlation with Outcome
  cor_with_outcome <- cor_matrix[feature_cols, "Outcome"]
  insights$linear_correlation <- list(
    values = round(cor_with_outcome, 3),
    strongest = names(which.max(abs(cor_with_outcome))),
    strongest_value = round(max(abs(cor_with_outcome)), 3),
    weakest = names(which.min(abs(cor_with_outcome))),
    weakest_value = round(min(abs(cor_with_outcome)), 3)
  )

  # 5. Non-linear correlation (Spearman)
  spearman_with_outcome <- cor_matrix_spearman[feature_cols, "Outcome"]
  insights$spearman_correlation <- list(
    values = round(spearman_with_outcome, 3),
    strongest = names(which.max(abs(spearman_with_outcome))),
    strongest_value = round(max(abs(spearman_with_outcome)), 3)
  )

  # 6. Mutual Information (non-linear dependency)
  insights$mutual_info <- list(
    top_feature = mi_df$Feature[1],
    top_mi = round(mi_df$MI[1], 4),
    top_3 = mi_df$Feature[1:3]
  )

  # 7. Non-linearity detection (Pearson vs Spearman difference)
  pearson_outcome <- cor_with_outcome
  spearman_outcome <- spearman_with_outcome
  diff_outcome <- abs(pearson_outcome - spearman_outcome)
  nonlinear_features <- names(diff_outcome[diff_outcome > 0.03])
  insights$nonlinearity <- list(
    features_with_nonlinear_relationship = nonlinear_features,
    has_nonlinear = length(nonlinear_features) > 0
  )

  # 8. Feature statistics by class
  class_diff <- sapply(feature_cols, function(col) {
    mean_pos <- mean(data[[col]][data$Outcome == 1])
    mean_neg <- mean(data[[col]][data$Outcome == 0])
    return((mean_pos - mean_neg) / sd(data[[col]]))  # Cohen's d approximation
  })
  insights$effect_size <- list(
    cohens_d = round(class_diff, 3),
    largest_effect = names(which.max(abs(class_diff))),
    largest_effect_value = round(max(abs(class_diff)), 3)
  )

  # 9. Multicollinearity check (features with high correlation)
  high_cor_pairs <- list()
  for (i in 1:(length(feature_cols) - 1)) {
    for (j in (i + 1):length(feature_cols)) {
      if (abs(cor_matrix[feature_cols[i], feature_cols[j]]) > 0.5) {
        high_cor_pairs[[length(high_cor_pairs) + 1]] <- list(
          var1 = feature_cols[i],
          var2 = feature_cols[j],
          correlation = round(cor_matrix[feature_cols[i], feature_cols[j]], 3)
        )
      }
    }
  }
  insights$multicollinearity <- list(
    high_cor_pairs = high_cor_pairs,
    has_multicollinearity = length(high_cor_pairs) > 0
  )

  return(insights)
}

# Generate insights
eda_insights <- generate_eda_insights(diabetes, feature_cols, mi_df, cor_matrix, cor_matrix_spearman)

# Print dynamic insights
cat("\n")
cat("=============================================================================\n")
cat("                    KEY INSIGHTS FROM EDA (Data-Driven)                      \n")
cat("=============================================================================\n\n")

cat("1. TARGET VARIABLE (Outcome):\n")
cat("   - No Diabetes (0):", eda_insights$class_balance$negative, "\n")
cat("   - Diabetes (1):", eda_insights$class_balance$positive, "\n")
cat("   - Imbalance ratio:", eda_insights$class_balance$imbalance_ratio, ": 1\n")
cat("   - Positive rate:", eda_insights$class_balance$positive_rate, "%\n")

cat("\n2. MISSING/ZERO VALUES (biologically impossible):\n")
for (i in seq_along(eda_insights$zero_values$columns)) {
  col <- eda_insights$zero_values$columns[i]
  cnt <- eda_insights$zero_values$counts[i]
  pct <- round(cnt / nrow(diabetes) * 100, 1)
  cat("   -", col, ":", cnt, "zeros (", pct, "%)\n")
}
cat("   - Worst:", eda_insights$zero_values$worst_column,
    "(", eda_insights$zero_values$worst_pct, "%)\n")

cat("\n3. SKEWNESS:\n")
for (col in feature_cols) {
  skew_val <- eda_insights$skewness$values[col]
  skew_type <- ifelse(abs(skew_val) > 1, "HIGHLY SKEWED",
                      ifelse(abs(skew_val) > 0.5, "moderately skewed", "normal"))
  cat("   -", col, ":", skew_val, "(", skew_type, ")\n")
}

cat("\n4. LINEAR CORRELATION WITH OUTCOME (Pearson):\n")
cor_sorted <- sort(abs(eda_insights$linear_correlation$values), decreasing = TRUE)
for (col in names(cor_sorted)) {
  cat("   -", col, ":", eda_insights$linear_correlation$values[col], "\n")
}
cat("   - Strongest:", eda_insights$linear_correlation$strongest,
    "(r =", eda_insights$linear_correlation$strongest_value, ")\n")

cat("\n5. NON-LINEAR CORRELATION (Spearman):\n")
cat("   - Strongest:", eda_insights$spearman_correlation$strongest,
    "(rho =", eda_insights$spearman_correlation$strongest_value, ")\n")

cat("\n6. MUTUAL INFORMATION (Non-Linear Dependency):\n")
cat("   - Top predictor:", eda_insights$mutual_info$top_feature,
    "(MI =", eda_insights$mutual_info$top_mi, "bits)\n")
cat("   - Top 3:", paste(eda_insights$mutual_info$top_3, collapse = ", "), "\n")

cat("\n7. NON-LINEARITY DETECTION:\n")
if (eda_insights$nonlinearity$has_nonlinear) {
  cat("   - Features with non-linear relationships:",
      paste(eda_insights$nonlinearity$features_with_nonlinear_relationship, collapse = ", "), "\n")
} else {
  cat("   - No strong non-linear relationships detected\n")
}

cat("\n8. EFFECT SIZE (Cohen's d approximation):\n")
cat("   - Largest effect:", eda_insights$effect_size$largest_effect,
    "(d =", eda_insights$effect_size$largest_effect_value, ")\n")

cat("\n9. MULTICOLLINEARITY:\n")
if (eda_insights$multicollinearity$has_multicollinearity) {
  for (pair in eda_insights$multicollinearity$high_cor_pairs) {
    cat("   -", pair$var1, "&", pair$var2, ": r =", pair$correlation, "\n")
  }
} else {
  cat("   - No high multicollinearity detected (all |r| < 0.5)\n")
}

cat("\n=============================================================================\n\n")

# -----------------------------------------------------------------------------
# 5. Data Preprocessing
# -----------------------------------------------------------------------------

df <- diabetes

# 5.1 Handle zero values (replace with median for biologically impossible zeros)
cat("=== Handling Zero Values ===\n")
zero_cols <- c("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
for (col in zero_cols) {
  n_zeros <- sum(df[[col]] == 0)
  if (n_zeros > 0) {
    median_val <- median(df[[col]][df[[col]] != 0])
    df[[col]][df[[col]] == 0] <- median_val
    cat("Replaced", n_zeros, "zeros in", col, "with median:", round(median_val, 2), "\n")
  }
}

# 5.2 Feature scaling (standardization for SVM and regularized models)
# Store original for tree-based models
df_original <- df

# Standardize features
preprocess_params <- preProcess(df[, feature_cols], method = c("center", "scale"))
df_scaled <- predict(preprocess_params, df)

cat("\n=== Preprocessed Data Structure ===\n")
str(df_scaled)

# -----------------------------------------------------------------------------
# 6. Split Data: Train (60%), Validation (20%), Test (20%)
# -----------------------------------------------------------------------------

n <- nrow(df)
train_idx <- sample(1:n, size = 0.6 * n)
remaining <- setdiff(1:n, train_idx)
val_idx <- sample(remaining, size = 0.5 * length(remaining))
test_idx <- setdiff(remaining, val_idx)

# Scaled data splits
train_data <- df_scaled[train_idx, ]
val_data <- df_scaled[val_idx, ]
test_data <- df_scaled[test_idx, ]

# Original data splits (for tree-based models)
train_data_orig <- df_original[train_idx, ]
val_data_orig <- df_original[val_idx, ]
test_data_orig <- df_original[test_idx, ]

cat("\n=== Data Split ===\n")
cat("Training set:", nrow(train_data), "samples (", round(nrow(train_data)/n*100, 1), "%)\n")
cat("Validation set:", nrow(val_data), "samples (", round(nrow(val_data)/n*100, 1), "%)\n")
cat("Test set:", nrow(test_data), "samples (", round(nrow(test_data)/n*100, 1), "%)\n")

# Check class distribution in splits
cat("\nClass distribution in training:", table(train_data$Outcome), "\n")
cat("Class distribution in validation:", table(val_data$Outcome), "\n")
cat("Class distribution in test:", table(test_data$Outcome), "\n")

X_train <- train_data %>% select(-Outcome)
y_train <- train_data$Outcome

X_val <- val_data %>% select(-Outcome)
y_val <- val_data$Outcome

X_test <- test_data %>% select(-Outcome)
y_test <- test_data$Outcome

# -----------------------------------------------------------------------------
# 7. Evaluation Metrics Function for Classification
# -----------------------------------------------------------------------------

evaluate_classification <- function(actual, predicted, predicted_prob, model_name) {
  # Confusion matrix
  cm <- confusionMatrix(factor(predicted, levels = c(0, 1)),
                        factor(actual, levels = c(0, 1)),
                        positive = "1")

  accuracy <- cm$overall["Accuracy"]
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Recall"]  # Sensitivity
  specificity <- cm$byClass["Specificity"]
  f1 <- cm$byClass["F1"]

  # AUC-ROC
  roc_obj <- roc(actual, predicted_prob, quiet = TRUE)
  auc_val <- auc(roc_obj)

  # AUC-PR (important for imbalanced data)
  pr_obj <- pr.curve(scores.class0 = predicted_prob[actual == 1],
                     scores.class1 = predicted_prob[actual == 0],
                     curve = FALSE)
  auc_pr <- pr_obj$auc.integral

  cat("\n=== ", model_name, " Performance ===\n")
  cat("Accuracy:", round(accuracy, 4), "\n")
  cat("Precision:", round(precision, 4), "\n")
  cat("Recall (Sensitivity):", round(recall, 4), "\n")
  cat("Specificity:", round(specificity, 4), "\n")
  cat("F1 Score:", round(f1, 4), "\n")
  cat("AUC-ROC:", round(auc_val, 4), "\n")
  cat("AUC-PR:", round(auc_pr, 4), "\n")

  return(list(
    metrics = data.frame(
      Model = model_name,
      Accuracy = as.numeric(accuracy),
      Precision = as.numeric(precision),
      Recall = as.numeric(recall),
      Specificity = as.numeric(specificity),
      F1 = as.numeric(f1),
      AUC_ROC = as.numeric(auc_val),
      AUC_PR = as.numeric(auc_pr)
    ),
    roc_obj = roc_obj,
    confusion_matrix = cm
  ))
}

# -----------------------------------------------------------------------------
# 8. Model 1: Logistic Regression with L2 Regularization (Ridge)
# -----------------------------------------------------------------------------
cat("\n")
cat("=============================================================================\n")
cat("          MODEL 1: LOGISTIC REGRESSION (L2 Regularization - Ridge)           \n")
cat("=============================================================================\n")

set.seed(42)

X_train_matrix <- as.matrix(X_train)
X_val_matrix <- as.matrix(X_val)
X_test_matrix <- as.matrix(X_test)

# Cross-validation to find optimal lambda
lr_cv <- cv.glmnet(
  x = X_train_matrix,
  y = y_train,
  family = "binomial",
  alpha = 0,  # Ridge
  nfolds = 10,
  type.measure = "auc"
)

cat("\nOptimal Lambda (min):", round(lr_cv$lambda.min, 6), "\n")
cat("Optimal Lambda (1se):", round(lr_cv$lambda.1se, 6), "\n")

plot(lr_cv, main = "Logistic Regression - Cross-Validation")

lr_model <- glmnet(
  x = X_train_matrix,
  y = y_train,
  family = "binomial",
  alpha = 0,
  lambda = lr_cv$lambda.min
)

# Coefficients
lr_coef <- coef(lr_model)
lr_coef_df <- data.frame(
  Feature = rownames(lr_coef),
  Coefficient = as.vector(lr_coef)
) %>%
  filter(Feature != "(Intercept)") %>%
  arrange(desc(abs(Coefficient)))

cat("\n=== Logistic Regression Coefficients ===\n")
print(lr_coef_df)

# Predictions on validation
lr_val_prob <- predict(lr_model, X_val_matrix, s = lr_cv$lambda.min, type = "response")[, 1]
lr_val_pred <- ifelse(lr_val_prob > 0.5, 1, 0)
lr_val_results <- evaluate_classification(y_val, lr_val_pred, lr_val_prob, "Logistic Regression (Val)")

# -----------------------------------------------------------------------------
# 9. Model 2: Random Forest (Bagging with Feature Sampling)
# -----------------------------------------------------------------------------
cat("\n")
cat("=============================================================================\n")
cat("            MODEL 2: RANDOM FOREST (Bagging + Feature Sampling)              \n")
cat("=============================================================================\n")

set.seed(42)

# Use original (non-scaled) data for tree-based models
rf_grid <- expand.grid(
  ntree = c(100, 200, 500),
  mtry = c(2, 3, 4)
)

best_rf_auc <- 0
best_rf_model <- NULL
best_rf_params <- NULL

cat("\nTuning Random Forest hyperparameters...\n")

for (i in 1:nrow(rf_grid)) {
  set.seed(42)
  rf_temp <- randomForest(
    x = train_data_orig[, feature_cols],
    y = factor(train_data_orig$Outcome),
    ntree = rf_grid$ntree[i],
    mtry = rf_grid$mtry[i],
    importance = TRUE
  )

  pred_prob <- predict(rf_temp, val_data_orig[, feature_cols], type = "prob")[, 2]
  roc_temp <- roc(val_data_orig$Outcome, pred_prob, quiet = TRUE)
  auc_temp <- auc(roc_temp)

  if (auc_temp > best_rf_auc) {
    best_rf_auc <- auc_temp
    best_rf_model <- rf_temp
    best_rf_params <- rf_grid[i, ]
  }
}

cat("\nBest Random Forest Parameters:\n")
cat("  ntree:", best_rf_params$ntree, "\n")
cat("  mtry:", best_rf_params$mtry, "\n")
cat("  Validation AUC:", round(best_rf_auc, 4), "\n")

# Feature Importance
rf_importance <- data.frame(
  Feature = rownames(importance(best_rf_model)),
  MeanDecreaseGini = importance(best_rf_model)[, "MeanDecreaseGini"]
) %>%
  arrange(desc(MeanDecreaseGini))

cat("\n=== Feature Importance (Random Forest) ===\n")
print(rf_importance)

ggplot(rf_importance, aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
  coord_flip() +
  labs(title = "Random Forest - Feature Importance",
       x = "Feature", y = "Mean Decrease in Gini") +
  theme_minimal()

# Validation predictions
rf_val_prob <- predict(best_rf_model, val_data_orig[, feature_cols], type = "prob")[, 2]
rf_val_pred <- ifelse(rf_val_prob > 0.5, 1, 0)
rf_val_results <- evaluate_classification(val_data_orig$Outcome, rf_val_pred, rf_val_prob, "Random Forest (Val)")

# -----------------------------------------------------------------------------
# 10. Model 3: Support Vector Machine (SVM) with RBF Kernel
# -----------------------------------------------------------------------------
cat("\n")
cat("=============================================================================\n")
cat("              MODEL 3: SVM (RBF Kernel - Non-Linear)                         \n")
cat("=============================================================================\n")

set.seed(42)

# Grid search for SVM hyperparameters
svm_grid <- expand.grid(
  cost = c(0.1, 1, 10),
  gamma = c(0.01, 0.1, 1)
)

best_svm_auc <- 0
best_svm_model <- NULL
best_svm_params <- NULL

cat("\nTuning SVM hyperparameters...\n")

for (i in 1:nrow(svm_grid)) {
  set.seed(42)
  svm_temp <- svm(
    x = as.matrix(X_train),
    y = factor(y_train),
    kernel = "radial",
    cost = svm_grid$cost[i],
    gamma = svm_grid$gamma[i],
    probability = TRUE
  )

  pred_prob <- attr(predict(svm_temp, as.matrix(X_val), probability = TRUE), "probabilities")[, "1"]
  roc_temp <- roc(y_val, pred_prob, quiet = TRUE)
  auc_temp <- auc(roc_temp)

  if (auc_temp > best_svm_auc) {
    best_svm_auc <- auc_temp
    best_svm_model <- svm_temp
    best_svm_params <- svm_grid[i, ]
  }
}

cat("\nBest SVM Parameters:\n")
cat("  Cost:", best_svm_params$cost, "\n")
cat("  Gamma:", best_svm_params$gamma, "\n")
cat("  Validation AUC:", round(best_svm_auc, 4), "\n")

# Validation predictions
svm_val_prob <- attr(predict(best_svm_model, as.matrix(X_val), probability = TRUE), "probabilities")[, "1"]
svm_val_pred <- ifelse(svm_val_prob > 0.5, 1, 0)
svm_val_results <- evaluate_classification(y_val, svm_val_pred, svm_val_prob, "SVM RBF (Val)")

# -----------------------------------------------------------------------------
# 11. Model 4: XGBoost (Gradient Boosting)
# -----------------------------------------------------------------------------
cat("\n")
cat("=============================================================================\n")
cat("                    MODEL 4: XGBOOST (Gradient Boosting)                     \n")
cat("=============================================================================\n")

set.seed(42)

# Prepare data for XGBoost - convert outcome to factor for classification
y_train_factor <- factor(train_data_orig$Outcome)
y_val_factor <- factor(val_data_orig$Outcome)

# Grid search for XGBoost
xgb_grid <- expand.grid(
  max_depth = c(3, 5, 7),
  learning_rate = c(0.01, 0.1, 0.3),
  subsample = c(0.7, 1.0)
)

best_xgb_auc <- 0
best_xgb_model <- NULL
best_xgb_params <- NULL
best_xgb_nrounds <- NULL

cat("\nTuning XGBoost hyperparameters...\n")

for (i in 1:nrow(xgb_grid)) {
  set.seed(42)

  # Use xgboost() with new API - factor y for classification
  xgb_temp <- xgboost(
    x = as.matrix(train_data_orig[, feature_cols]),
    y = y_train_factor,
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = xgb_grid$max_depth[i],
    learning_rate = xgb_grid$learning_rate[i],
    subsample = xgb_grid$subsample[i],
    colsample_bytree = 0.8,
    nrounds = 100,
    print_every_n = 0
  )

  pred_prob <- predict(xgb_temp, as.matrix(val_data_orig[, feature_cols]), type = "response")
  roc_temp <- roc(val_data_orig$Outcome, pred_prob, quiet = TRUE)
  auc_temp <- auc(roc_temp)

  if (auc_temp > best_xgb_auc) {
    best_xgb_auc <- auc_temp
    best_xgb_model <- xgb_temp
    best_xgb_params <- xgb_grid[i, ]
    best_xgb_nrounds <- 100
  }
}

cat("\nBest XGBoost Parameters:\n")
cat("  max_depth:", best_xgb_params$max_depth, "\n")
cat("  learning_rate:", best_xgb_params$learning_rate, "\n")
cat("  subsample:", best_xgb_params$subsample, "\n")
cat("  nrounds:", best_xgb_nrounds, "\n")
cat("  Validation AUC:", round(best_xgb_auc, 4), "\n")

# Feature Importance
xgb_importance <- xgb.importance(model = best_xgb_model)
cat("\n=== Feature Importance (XGBoost) ===\n")
print(xgb_importance)

ggplot(xgb_importance, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity", fill = "darkgreen", alpha = 0.7) +
  coord_flip() +
  labs(title = "XGBoost - Feature Importance (Gain)",
       x = "Feature", y = "Gain") +
  theme_minimal()

# Validation predictions
xgb_val_prob <- predict(best_xgb_model, as.matrix(val_data_orig[, feature_cols]), type = "response")
xgb_val_pred <- ifelse(xgb_val_prob > 0.5, 1, 0)
xgb_val_results <- evaluate_classification(val_data_orig$Outcome, xgb_val_pred, xgb_val_prob, "XGBoost (Val)")

# -----------------------------------------------------------------------------
# 12. Model Comparison on Validation Set
# -----------------------------------------------------------------------------
cat("\n")
cat("=============================================================================\n")
cat("                    MODEL COMPARISON (VALIDATION SET)                        \n")
cat("=============================================================================\n")

comparison_val <- rbind(
  lr_val_results$metrics,
  rf_val_results$metrics,
  svm_val_results$metrics,
  xgb_val_results$metrics
)
print(comparison_val)

# ROC Curves comparison
plot(lr_val_results$roc_obj, col = "blue", main = "ROC Curves - Validation Set")
plot(rf_val_results$roc_obj, col = "red", add = TRUE)
plot(svm_val_results$roc_obj, col = "green", add = TRUE)
plot(xgb_val_results$roc_obj, col = "purple", add = TRUE)
legend("bottomright",
       legend = c(paste("Logistic Regression (AUC:", round(auc(lr_val_results$roc_obj), 3), ")"),
                  paste("Random Forest (AUC:", round(auc(rf_val_results$roc_obj), 3), ")"),
                  paste("SVM RBF (AUC:", round(auc(svm_val_results$roc_obj), 3), ")"),
                  paste("XGBoost (AUC:", round(auc(xgb_val_results$roc_obj), 3), ")")),
       col = c("blue", "red", "green", "purple"), lwd = 2)

# Bar plot comparison
comparison_long <- comparison_val %>%
  pivot_longer(cols = c(Accuracy, Precision, Recall, F1, AUC_ROC, AUC_PR),
               names_to = "Metric", values_to = "Value")

ggplot(comparison_long, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
  labs(title = "Model Comparison on Validation Set") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "bottom")

# -----------------------------------------------------------------------------
# 13. Final Evaluation on Test Set
# -----------------------------------------------------------------------------
cat("\n")
cat("=============================================================================\n")
cat("                     FINAL EVALUATION (TEST SET)                             \n")
cat("=============================================================================\n")

# Logistic Regression
lr_test_prob <- predict(lr_model, X_test_matrix, s = lr_cv$lambda.min, type = "response")[, 1]
lr_test_pred <- ifelse(lr_test_prob > 0.5, 1, 0)
lr_test_results <- evaluate_classification(y_test, lr_test_pred, lr_test_prob, "Logistic Regression (Test)")

# Random Forest
rf_test_prob <- predict(best_rf_model, test_data_orig[, feature_cols], type = "prob")[, 2]
rf_test_pred <- ifelse(rf_test_prob > 0.5, 1, 0)
rf_test_results <- evaluate_classification(test_data_orig$Outcome, rf_test_pred, rf_test_prob, "Random Forest (Test)")

# SVM
svm_test_prob <- attr(predict(best_svm_model, X_test_matrix, probability = TRUE), "probabilities")[, "1"]
svm_test_pred <- ifelse(svm_test_prob > 0.5, 1, 0)
svm_test_results <- evaluate_classification(y_test, svm_test_pred, svm_test_prob, "SVM RBF (Test)")

# XGBoost
xgb_test_prob <- predict(best_xgb_model, as.matrix(test_data_orig[, feature_cols]), type = "response")
xgb_test_pred <- ifelse(xgb_test_prob > 0.5, 1, 0)
xgb_test_results <- evaluate_classification(test_data_orig$Outcome, xgb_test_pred, xgb_test_prob, "XGBoost (Test)")

# -----------------------------------------------------------------------------
# 14. DYNAMIC Final Conclusion (4 Models)
# -----------------------------------------------------------------------------

comparison_test <- rbind(
  lr_test_results$metrics,
  rf_test_results$metrics,
  svm_test_results$metrics,
  xgb_test_results$metrics
)

cat("\n")
cat("=============================================================================\n")
cat("                        FINAL MODEL COMPARISON                               \n")
cat("=============================================================================\n")
print(comparison_test)

# Rank models by key metrics for classification
comparison_test$AUC_rank <- rank(-comparison_test$AUC_ROC)
comparison_test$F1_rank <- rank(-comparison_test$F1)
comparison_test$Recall_rank <- rank(-comparison_test$Recall)
comparison_test$Precision_rank <- rank(-comparison_test$Precision)
comparison_test$Total_score <- comparison_test$AUC_rank + comparison_test$F1_rank +
                               comparison_test$Recall_rank + comparison_test$Precision_rank

# Find winner
comparison_test_sorted <- comparison_test %>% arrange(Total_score)
winner <- comparison_test_sorted$Model[1]
winner_auc <- comparison_test_sorted$AUC_ROC[1]
winner_f1 <- comparison_test_sorted$F1[1]
winner_recall <- comparison_test_sorted$Recall[1]
winner_precision <- comparison_test_sorted$Precision[1]

second_best <- comparison_test_sorted$Model[2]
second_auc <- comparison_test_sorted$AUC_ROC[2]

# Model categories for analysis
linear_models <- c("Logistic Regression (Test)")
nonlinear_models <- c("Random Forest (Test)", "SVM RBF (Test)", "XGBoost (Test)")
bagging_models <- c("Random Forest (Test)")
boosting_models <- c("XGBoost (Test)")

# Analysis
linear_auc <- comparison_test$AUC_ROC[comparison_test$Model %in% linear_models]
nonlinear_avg_auc <- mean(comparison_test$AUC_ROC[comparison_test$Model %in% nonlinear_models])
bagging_auc <- comparison_test$AUC_ROC[comparison_test$Model %in% bagging_models]
boosting_auc <- comparison_test$AUC_ROC[comparison_test$Model %in% boosting_models]

# Top predictors from different models
top_rf_features <- head(rf_importance$Feature, 3)
top_xgb_features <- head(xgb_importance$Feature, 3)
top_lr_features <- head(lr_coef_df$Feature, 3)

cat("\n")
cat("=============================================================================\n")
cat("                         CONCLUSION (Data-Driven)                            \n")
cat("=============================================================================\n\n")

cat("MODEL RANKING (by combined metrics - lower is better):\n")
for (i in 1:nrow(comparison_test_sorted)) {
  cat("  ", i, ". ", comparison_test_sorted$Model[i],
      " (AUC: ", round(comparison_test_sorted$AUC_ROC[i], 4),
      ", F1: ", round(comparison_test_sorted$F1[i], 4), ")\n", sep = "")
}

cat("\n")
cat("BEST MODEL:", winner, "\n")
cat("  - AUC-ROC:", round(winner_auc, 4), "\n")
cat("  - F1 Score:", round(winner_f1, 4), "\n")
cat("  - Recall:", round(winner_recall, 4), "\n")
cat("  - Precision:", round(winner_precision, 4), "\n")
cat("  - Outperforms", second_best, "by",
    round((winner_auc - second_auc) * 100, 2), "percentage points (AUC)\n")

cat("\n")
cat("-----------------------------------------------------------------------------\n")
cat("                          KEY FINDINGS                                       \n")
cat("-----------------------------------------------------------------------------\n\n")

# Finding 1: Linear vs Non-linear
cat("1. LINEAR vs NON-LINEAR MODELS:\n")
if (nonlinear_avg_auc > linear_auc) {
  cat("   - Non-linear models (avg AUC:", round(nonlinear_avg_auc, 4),
      ") OUTPERFORM linear (AUC:", round(linear_auc, 4), ")\n")
  cat("   - Suggests non-linear decision boundaries in data\n\n")
} else {
  cat("   - Linear model (AUC:", round(linear_auc, 4),
      ") competitive with non-linear (avg AUC:", round(nonlinear_avg_auc, 4), ")\n")
  cat("   - Relationships may be largely linear\n\n")
}

# Finding 2: Bagging vs Boosting
cat("2. BAGGING vs BOOSTING:\n")
cat("   - Random Forest (Bagging) AUC:", round(bagging_auc, 4), "\n")
cat("   - XGBoost (Boosting) AUC:", round(boosting_auc, 4), "\n")
if (boosting_auc > bagging_auc) {
  cat("   - Boosting performs better - sequential error correction helps\n\n")
} else {
  cat("   - Bagging performs better - variance reduction more important here\n\n")
}

# Finding 3: Class imbalance impact
cat("3. CLASS IMBALANCE:\n")
cat("   - Imbalance ratio:", eda_insights$class_balance$imbalance_ratio, ": 1\n")
cat("   - Positive class rate:", eda_insights$class_balance$positive_rate, "%\n")
if (eda_insights$class_balance$imbalance_ratio > 2) {
  cat("   - Consider: SMOTE, class weights, or threshold tuning for production\n\n")
} else {
  cat("   - Moderate imbalance - standard approaches should work\n\n")
}

# Finding 4: Top predictors comparison
cat("4. TOP PREDICTORS ACROSS MODELS:\n")
cat("   - Random Forest:", paste(top_rf_features, collapse = ", "), "\n")
cat("   - XGBoost:", paste(top_xgb_features, collapse = ", "), "\n")
cat("   - Logistic Regression:", paste(top_lr_features, collapse = ", "), "\n")

all_top1 <- c(top_rf_features[1], top_xgb_features[1], top_lr_features[1])
if (length(unique(all_top1)) == 1) {
  cat("   - All models agree:", unique(all_top1), "is most important\n\n")
} else {
  cat("   - Models show different importance rankings\n\n")
}

# Finding 5: Non-linearity from EDA
cat("5. NON-LINEARITY (from EDA):\n")
if (eda_insights$nonlinearity$has_nonlinear) {
  cat("   - Non-linear relationships detected in:",
      paste(eda_insights$nonlinearity$features_with_nonlinear_relationship, collapse = ", "), "\n")
  cat("   - This explains why non-linear models may perform better\n\n")
} else {
  cat("   - No strong non-linear relationships detected\n\n")
}

# Finding 6: Data quality impact
cat("6. DATA QUALITY:\n")
cat("   - Zero values replaced with medians for:",
    paste(eda_insights$zero_values$columns, collapse = ", "), "\n")
cat("   - Most affected:", eda_insights$zero_values$worst_column,
    "(", eda_insights$zero_values$worst_pct, "% zeros)\n")
if (eda_insights$multicollinearity$has_multicollinearity) {
  cat("   - Multicollinearity present - regularization helps\n")
}

cat("\n=============================================================================\n")

# -----------------------------------------------------------------------------
# 15. Visualization
# -----------------------------------------------------------------------------

# Final ROC curves
plot(lr_test_results$roc_obj, col = "blue", main = "ROC Curves - Test Set", lwd = 2)
plot(rf_test_results$roc_obj, col = "red", add = TRUE, lwd = 2)
plot(svm_test_results$roc_obj, col = "green", add = TRUE, lwd = 2)
plot(xgb_test_results$roc_obj, col = "purple", add = TRUE, lwd = 2)
legend("bottomright",
       legend = c(paste("Logistic Regression (AUC:", round(auc(lr_test_results$roc_obj), 3), ")"),
                  paste("Random Forest (AUC:", round(auc(rf_test_results$roc_obj), 3), ")"),
                  paste("SVM RBF (AUC:", round(auc(svm_test_results$roc_obj), 3), ")"),
                  paste("XGBoost (AUC:", round(auc(xgb_test_results$roc_obj), 3), ")")),
       col = c("blue", "red", "green", "purple"), lwd = 2)

# Confusion matrices visualization
cat("\n=== Confusion Matrix - Best Model ===\n")
if (grepl("Logistic", winner)) {
  print(lr_test_results$confusion_matrix)
} else if (grepl("Random Forest", winner)) {
  print(rf_test_results$confusion_matrix)
} else if (grepl("SVM", winner)) {
  print(svm_test_results$confusion_matrix)
} else {
  print(xgb_test_results$confusion_matrix)
}

# Final comparison bar chart
comparison_test_clean <- comparison_test %>%
  select(Model, Accuracy, Precision, Recall, F1, AUC_ROC, AUC_PR)

comparison_final_long <- comparison_test_clean %>%
  pivot_longer(cols = c(Accuracy, Precision, Recall, F1, AUC_ROC, AUC_PR),
               names_to = "Metric", values_to = "Value")

ggplot(comparison_final_long, aes(x = reorder(Model, -Value), y = Value, fill = Model)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  facet_wrap(~Metric, scales = "free") +
  labs(title = "Final Model Comparison (Test Set)",
       x = "", y = "Value") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

# Feature importance comparison
cat("\n=== Feature Importance Summary ===\n")
importance_summary <- data.frame(
  Feature = feature_cols,
  RF_Rank = match(feature_cols, rf_importance$Feature),
  XGB_Rank = match(feature_cols, xgb_importance$Feature),
  LR_Rank = match(feature_cols, lr_coef_df$Feature),
  MI_Rank = match(feature_cols, mi_df$Feature)
)
importance_summary$Avg_Rank <- rowMeans(importance_summary[, c("RF_Rank", "XGB_Rank", "LR_Rank", "MI_Rank")], na.rm = TRUE)
importance_summary <- importance_summary %>% arrange(Avg_Rank)
print(importance_summary)
