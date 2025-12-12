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

# Préparer les données
X <- X.reg %>% dplyr::select(-y) %>% as.matrix()
y <- X.reg$y
X_scaled <- scale(X)

# Grille fine de lambda
lambda_grid <- 10^seq(-2, 2, length.out = 200)

# Explorer alpha proche de LASSO
alpha_grid <- seq(0.8, 1, by = 0.02)

best_mse <- Inf
best_alpha <- NA
best_lambda <- NA
best_model <- NULL

for(a in alpha_grid){
  en_mod <- cv.glmnet(X_scaled, y, alpha = a, nfolds = 10, lambda = lambda_grid)
  mse <- min(en_mod$cvm)
  if(mse < best_mse){
    best_mse <- mse
    best_alpha <- a
    best_lambda <- en_mod$lambda.min
    best_model <- en_mod
  }
}

cat("Meilleur alpha :", best_alpha, "\n")
cat("Lambda correspondant :", best_lambda, "\n")
cat("MSE CV (d'après cv.glmnet) :", best_mse, "\n\n")

# Variables sélectionnées
coef_selected <- coef(best_model, s = "lambda.min")
selected_vars <- rownames(coef_selected)[coef_selected[,1] != 0]
selected_vars <- selected_vars[selected_vars != "(Intercept)"]
cat("Variables sélectionnées :", selected_vars, "\n\n")

formula_bam <- paste(selected_vars,collapse=")+s(")
final_formula <- paste0("y~s(",formula_bam,")")
cat("Formule finale sélectionnée: ",final_formula)

df <- data.frame(y = y, X_scaled[,selected_vars])
colnames(df)[-1] <- selected_vars

#fit avec BAM

reg<-bam(as.formula(final_formula),family=gaussian(),data=df)

# Fonction regresseur pour la plateforme
regresseur <- function(test_set) {
  library(mgcv)
  # Convertir en dataframe et rescale avec les variables sélectionnées.
  X_test_scaled <- scale(test_set)
  X_test <- data.frame(X_test_scaled)[,selected_vars]
  colnames(X_test)[-length(X_test)] <- selected_vars
  # Prédiction
  preds <- as.numeric(predict.bam(reg,newdata = X_test))
  return(preds)
}

# =========================================
# 4. Sauvegarder l'environnement minimal
save(
  clas,
  reg,
  classifieur,
  regresseur,
  X_mean_clas,
  X_sd_clas,
  selected_vars,
  file = "env.Rdata"
)

cat("Fichier env.Rdata créé avec succès !\n")
cat("Contenu :", ls()[ls() %in% c('clas','reg','classifieur','regresseur')], "\n")


