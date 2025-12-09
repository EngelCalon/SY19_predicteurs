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
# 3. Préparation régression linéaire avec AIC
#Formule de AIC
fit <- lm(y ~ ., data=X.reg)
sel.aic <- stepAIC(fit,scope=y ~ .,direction="both", trace=FALSE)
formula <- formula(sel.aic)

#Regression linéaire
reg <- lm(formula, data = X.reg)

# Fonction regresseur pour la plateforme
regresseur <- function(test_set) {
  library(MASS)
  # Prédiction
  preds <- as.numeric(predict(reg, newdata = test_set))
  return(preds)
}

# =========================================
# 4. Sauvegarder l'environnement minimal
save(clas, reg, classifieur, regresseur, file = "env.Rdata")

cat("Fichier env.Rdata créé avec succès !\n")
cat("Contenu :", ls()[ls() %in% c('clas','reg','classifieur','regresseur','X_mean','X_sd')], "\n")
