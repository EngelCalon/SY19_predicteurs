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

# Fonction classifieur
classifieur <- function(test_set) {
  library(MASS)
  preds <- predict(clas, test_set)$class
  return(preds)
}

# =========================================
# 3. Préparation régression LASSO simple

ytrain <- X.reg$y
X_reg_df <- X.reg[, setdiff(colnames(X.reg), "y"), drop = FALSE]

# Standardisation
xtrain <- scale(X_reg_df)

# On entraîne LASSO sur tout le jeu d'apprentissage
cv.out<-cv.glmnet(xtrain,ytrain,alpha=1)
reg <- glmnet(x = xtrain, y = ytrain,lambda=cv.out$lambda.min,alpha=1)

# On stocke la moyenne et l'écart type pour reproduire la standardisation sur le test
X_mean <- attr(xtrain, "scaled:center")
X_sd   <- attr(xtrain, "scaled:scale")

# Fonction regresseur pour la plateforme
regresseur <- function(test_set) {
  library(glmnet)
  # Convertir en matrice et standardiser avec les mêmes moyennes/écarts type
  X_test_mat <- as.matrix(test_set)
  X_test_scaled <- sweep(X_test_mat, 2, X_mean, FUN = "-")
  X_test_scaled <- sweep(X_test_scaled, 2, X_sd, FUN = "/")
  
  # Prédiction
  preds <- as.numeric(predict(reg, newx = X_test_scaled, s = cv.out$lambda.min))
  return(preds)
}

# =========================================
# 4. Sauvegarder l'environnement minimal
save(clas, reg, classifieur, regresseur, 
     X_mean, X_sd,  # pour reproduire la standardisation
     file = "env.Rdata")

cat("Fichier env.Rdata créé avec succès !\n")
cat("Contenu :", ls()[ls() %in% c('clas','reg','classifieur','regresseur','X_mean','X_sd')], "\n")

