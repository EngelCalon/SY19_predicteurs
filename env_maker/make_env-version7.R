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
# 3. Préparation régression linéaire avec AIC

#Préparation des données
ytrain <- X.reg$y
X_reg_df <- X.reg[, setdiff(colnames(X.reg), "y"), drop = FALSE]

# Standardisation
xtrain <- scale(X_reg_df)

#Dataframe après scale
df_train <- data.frame(y = ytrain, xtrain)

#Formule de AIC
fit <- lm(y ~ ., data=df_train)
sel.aic <- stepAIC(fit,scope=y ~ .,direction="both", trace=FALSE)
formula <- formula(sel.aic)

#Regression linéaire
reg <- lm(formula, data = df_train)

# On stocke la moyenne et l'écart type pour reproduire la standardisation sur le test
X_mean_reg <- attr(xtrain, "scaled:center")
X_sd_reg   <- attr(xtrain, "scaled:scale")

# Fonction regresseur pour la plateforme
regresseur <- function(test_set) {
  library(MASS)
  # Convertir en matrice et standardiser avec les mêmes moyennes/écarts type
  X_test_scaled <- sweep(as.matrix(test_set[, names(test_set) != "y"]), 2, X_mean_reg, "-")
  X_test_scaled <- sweep(X_test_scaled, 2, X_sd_reg, "/")
  X_test_scaled <- data.frame(X_test_scaled)
  # Prédiction
  preds <- as.numeric(predict(reg, newdata = X_test_scaled))
  return(preds)
}

# =========================================
# 4. Sauvegarder l'environnement minimal
save(clas, reg, classifieur, regresseur, 
     X_mean_clas, X_sd_clas,  # pour la classification
     X_mean_reg, X_sd_reg,            # pour la régression
     file = "env.Rdata")

cat("Fichier env.Rdata créé avec succès !\n")
cat("Contenu :", ls()[ls() %in% c('clas','reg','classifieur','regresseur','X_mean','X_sd')], "\n")
