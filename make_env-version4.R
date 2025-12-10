# Installer les packages nécessaires
packages_needed <- c("MASS", "glmnet", "dplyr", "here", "e1071")
to_install <- packages_needed[!(packages_needed %in% installed.packages()[, "Package"])]
if(length(to_install)) install.packages(to_install)

library(MASS)
library(glmnet)
library(dplyr)
library(here)
library(e1071)

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
# 2. Préparation classification
X.clas$y <- as.factor(X.clas$y)

# Garder uniquement X21:X50 pour la classification
vars_clas <- c(paste0("X",21:50), "y")
X_clas_sel <- X.clas[, vars_clas]

# Standardisation
X_train_scaled <- scale(X_clas_sel[, -ncol(X_clas_sel)])
X_train_scaled_df <- data.frame(X_train_scaled, y = X_clas_sel$y)
X_mean_clas <- attr(X_train_scaled, "scaled:center")
X_sd_clas   <- attr(X_train_scaled, "scaled:scale")

# Entraîner QDA 
qda_model <- qda(y ~ ., data = X_train_scaled_df)

# Entraîner SVM RBF
svm_model <- svm(y ~ ., data = X_train_scaled_df,
                 kernel = "radial",
                 gamma = 0.05,
                 cost  = 10)

# Fonction classifieur (vote majoritaire QDA + SVM)
classifieur <- function(test_set) {
  # Standardisation avec les mêmes paramètres
  X_test_scaled <- sweep(as.matrix(test_set), 2, X_mean_clas, FUN = "-")
  X_test_scaled <- sweep(X_test_scaled, 2, X_sd_clas,   FUN = "/")
  test_scaled_df <- data.frame(X_test_scaled)
  
  # Prédictions QDA
  pred_qda <- predict(qda_model, newdata = test_scaled_df)$class
  
  # Prédictions SVM
  pred_svm <- predict(svm_model, newdata = test_scaled_df)
  
  # Vote majoritaire
  pred_final <- as.factor(apply(cbind(as.character(pred_qda),
                                      as.character(pred_svm)),
                                1,
                                function(x) names(which.max(table(x)))))
  return(pred_final)
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
save(qda_model, svm_model, reg, classifieur, regresseur,
     X_mean, X_sd,X_mean_clas, X_sd_clas,
     file = "env.Rdata")

cat("Fichier env.Rdata créé avec succès !\n")
cat("Contenu :", ls()[ls() %in% c('qda_model','svm_model','reg','classifieur','regresseur','X_mean','X_sd')], "\n")
