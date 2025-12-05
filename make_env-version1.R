# Installer les packages nécessaires
packages_needed <- c("MASS", "glmnet", "dplyr", "here")
to_install <- packages_needed[!(packages_needed %in% installed.packages()[, "Package"])]
if(length(to_install)) install.packages(to_install)

library(MASS)
library(glmnet)
library(dplyr)
library(here)

# =========================================
# 1. Lecture des jeux d'apprentissage

# Modifier les chemins si nécessaire
clas_data_path <- here("src", "data", "TP5_a25_clas_app.txt")
reg_data_path <- here("src", "data", "TP5_a25_reg_app.txt")

# Lecture des données
X.reg <- read.table(
  reg_data_path,
  header = TRUE,
  sep = " ",
  quote = "\"",
  stringsAsFactors = FALSE
)
X.clas <- read.table(
  clas_data_path,
  header = TRUE,
  sep = " ",
  quote = "\"",
  stringsAsFactors = FALSE
)

# =========================================
# 2. Préparation classification => A REMPLACER
X.clas$y <- as.factor(X.clas$y)

clas <- MASS::lda(y ~ ., data = X.clas)

classifieur <- function(test_set) {
  library(MASS)
  preds <- predict(clas, test_set)$class
  return(preds)
}

# =========================================
# 3. Préparation régression Elastic Net

y_reg <- X.reg$y
X_reg_df <- X.reg[, setdiff(colnames(X.reg), "y"), drop = FALSE]

# Standardisation des données
X_scaled <- scale(X_reg_df)
X_mean <- attr(X_scaled, "scaled:center")
X_sd   <- attr(X_scaled, "scaled:scale")

# Grille fine de lambda et alpha
lambda_grid <- 10^seq(-2, 2, length.out = 200)
alpha_grid  <- seq(0.8, 1, by = 0.02)

# Paramètres de repeated CV
n_repeats <- 20           # nombre de répétitions avec différentes seeds
seeds <- sample(1:10000, n_repeats) # seeds différentes pour chaque répétition

# Stocker les résultats
results <- list()

for(a in alpha_grid){
  mse_seeds <- numeric(n_repeats)
  
  for(i in seq_along(seeds)){
    set.seed(seeds[i])
    cv_mod <- cv.glmnet(X_scaled, y_reg, alpha = a, nfolds = 10, lambda = lambda_grid)
    mse_seeds[i] <- min(cv_mod$cvm)
  }
  
  # Moyenne de la MSE sur toutes les seeds
  results[[as.character(a)]] <- mean(mse_seeds)
}

# Sélection du meilleur alpha
best_alpha <- as.numeric(names(results)[which.min(unlist(results))])
cat("Meilleur alpha moyen sur repeated CV :", best_alpha, "\n")
best_mse   <- results[[as.character(best_alpha)]]
cat("MSE CV moyen sur repeated CV :", best_mse, "\n")

# Fit avec alpha optimal pour obtenir le meilleur lambda
final_cv <- cv.glmnet(X_scaled, y_reg, alpha = best_alpha, nfolds = 10, lambda = lambda_grid)
best_lambda <- final_cv$lambda.min
cat("Lambda correspondant au meilleur alpha :", best_lambda, "\n")

# Variables sélectionnées
coef_selected <- coef(final_cv, s = "lambda.min")
selected_vars <- rownames(coef_selected)[coef_selected[,1] != 0]
selected_vars <- selected_vars[selected_vars != "(Intercept)"]
cat("Variables sélectionnées :", selected_vars, "\n\n")

# Refit final du modèle Elastic Net sur toutes les données
reg <- glmnet(X_scaled[, selected_vars, drop = FALSE], y_reg, alpha = best_alpha, lambda = best_lambda)

# Fonction regresseur pour la plateforme
regresseur <- function(test_set) {
  library(glmnet)
  X_test_mat <- as.matrix(test_set[, selected_vars, drop = FALSE])
  # Standardisation avec les mêmes moyennes et écarts-types
  X_test_scaled <- sweep(X_test_mat, 2, X_mean[selected_vars], FUN = "-")
  X_test_scaled <- sweep(X_test_scaled, 2, X_sd[selected_vars], FUN = "/")
  
  preds <- as.numeric(predict(reg, newx = X_test_scaled, s = best_lambda))
  return(preds)
}

# =========================================
# 4. Sauvegarder les objets essentiels
save(clas, reg, classifieur, regresseur, X_mean, X_sd, selected_vars,
     file = "env.Rdata")

cat("Fichier env.Rdata créé avec succès !\n")
cat("Contenu :", ls()[ls() %in% c('clas','reg','classifieur','regresseur','X_mean','X_sd','selected_vars', 'best_lambda')], "\n")
