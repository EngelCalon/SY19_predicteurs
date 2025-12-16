# Installer les packages nécessaires
packages_needed <- c("MASS", "glmnet", "dplyr", "here")
to_install <- packages_needed[!(packages_needed %in% installed.packages()[, "Package"])]
if(length(to_install)) install.packages(to_install)

library(MASS)
library(glmnet)
library(dplyr)
library(here)
library(mgcv)

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
# 2. Préparation classification (QDA simple sans X1 à X20)

# Cible en facteur
X.clas$y <- as.factor(X.clas$y)

# Sélection des variables X21:X50 + y
vars_clas <- c(paste0("X", 21:50), "y")
X_clas_sel <- X.clas[, vars_clas]

# Standardisation des X
X_clas_scaled_mat <- scale(X_clas_sel[, -ncol(X_clas_sel)])

# Sauvegarde des paramètres de standardisation
X_mean_clas <- attr(X_clas_scaled_mat, "scaled:center")
X_sd_clas   <- attr(X_clas_scaled_mat, "scaled:scale")

# Data frame final pour l'entraînement
X_clas_scaled <- data.frame(X_clas_scaled_mat, y = X_clas_sel$y)

# Entraînement QDA
clas <- qda(y ~ ., data = X_clas_scaled)

# Fonction classifieur pour la plateforme
classifieur <- function(test_set) {
  library(MASS)
  
  # Garder X21:X50
  test_sub <- test_set[, paste0("X", 21:50), drop = FALSE]
  
  # Standardisation avec les bons paramètres
  X_test_scaled <- sweep(as.matrix(test_sub), 2, X_mean_clas, FUN = "-")
  X_test_scaled <- sweep(X_test_scaled, 2, X_sd_clas, FUN = "/")
  
  test_scaled_df <- data.frame(X_test_scaled)
  
  # Prediction
  preds <- predict(clas, newdata = test_scaled_df)$class
  return(preds)
}

# =========================================
# 3. Préparation régression LASSO OPTI + BAM

# Préparation des données
X <- as.matrix(reg_data[, setdiff(names(reg_data), "y")])
y <- reg_data$y
X_scaled <- scale(X)

# LASSO pour la sélection
cv_lasso <- cv.glmnet(X_scaled, y, alpha = 1, nfolds = 10)
best_lambda <- cv_lasso$lambda.min

# Coefficients
lasso_coef <- coef(cv_lasso, s = best_lambda)
selected_vars <- rownames(lasso_coef)[lasso_coef[,1] != 0]
selected_vars <- selected_vars[selected_vars != "(Intercept)"]
cat("Variables sélectionnées par LASSO :\n")
print(selected_vars)

# 3. Construire un GAM/BAM sur les variables sélectionnées
X_scaled_df <- as.data.frame(X_scaled)
colnames(X_scaled_df) <- colnames(X)
df_gam <- data.frame(y = y, X_scaled_df[, selected_vars, drop = FALSE])

form_str <- paste0("y ~ ", paste0("s(", selected_vars, ", bs='ps', k=5)", collapse = " + "))
bam_model <- bam(as.formula(form_str), data = df_gam, family = gaussian())
# On stocke la moyenne et l'écart type pour reproduire la standardisation sur le test

X_mean_reg <- attr(X_scaled, "scaled:center")
X_sd_reg   <- attr(X_scaled, "scaled:scale")

# Fonction regresseur pour la plateforme
regresseur <- function(test_set) {
  library(mgcv)
  # Convertir en dataframe et rescale avec les variables sélectionnées.
  X_test_scaled <- sweep(as.matrix(test_set[,selected_vars]), 2, X_mean_reg[selected_vars], FUN="-")
  X_test_scaled <- sweep(X_test_scaled, 2, X_sd_reg[selected_vars], FUN="/")
  X_test <- data.frame(X_test_scaled)
  # Prédiction
  preds <- as.numeric(predict(bam_model, newdata = X_test))
  return(preds)
}

# =========================================
# 4. Sauvegarder l'environnement minimal
save(
  clas,
  bam_model,
  classifieur,
  regresseur,
  X_mean_reg,
  X_sd_reg,
  X_mean_clas,
  X_sd_clas,
  selected_vars,
  file = "env.Rdata"
)

cat("Fichier env.Rdata créé avec succès !\n")
cat("Contenu :", ls()[ls() %in% c('clas','reg','classifieur','regresseur')], "\n")


