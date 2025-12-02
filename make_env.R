# Installer les packages
packages_needed <- c("MASS", "glmnet")
to_install <- packages_needed[!(packages_needed %in% installed.packages()[, "Package"])]
if(length(to_install)) install.packages(to_install)

library(MASS)
library(glmnet)

######################################################
# 1. Lecture des jeux d'apprentissage

clas_path <- "a25_clas_app.txt"
reg_path  <- "a25_reg_app.txt"

X.clas <- read.table(clas_path, header = TRUE, stringsAsFactors = FALSE)
X.reg  <- read.table(reg_path,  header = TRUE, stringsAsFactors = FALSE)

# Ajouter des vérifications ??

######################################################
# 2. Préparation classification => A COMPLETER

# Convertir y en facteur (si ce n'est pas déjà le cas)
X.clas$y <- as.factor(X.clas$y)
# A COMPLETER

# Entraînement LDA (MASS)
clas <- MASS::lda(y ~ ., data = X.clas)

# Fonction classifieur => A COMPLETER
classifieur <- function(test_set) {
  # Charger les librairies nécessaires 
  library(MASS)
  preds <- predict(clas, test_set)$class
  return(preds)
}

###################################################
# 3. Préparation régression 

y_reg <- X.reg$y
X.reg.df <- X.reg[, setdiff(colnames(X.reg), "y"), drop = FALSE]

# Conversion en numérique => jsp si c'est vraiment utile...
X.reg.mat <- as.matrix(sapply(X.reg.df, function(col) {
  if(is.factor(col)) as.numeric(as.character(col)) else as.numeric(col)
}))

# Doit on ajouter une gestion des NA ? 

set.seed(123)
reg_cv <- cv.glmnet(x = X.reg.mat, y = y_reg, alpha = 1, nfolds = 10)  # LASSO (alpha=1)
reg <- reg_cv

# Définition de la fonction regresseur
regresseur <- function(test_set) {
  # Charger packages nécessaires
  library(glmnet)
  # Appliquer sur test les transformations appliquées sur le jeu d'entraînement
  X_test_mat <- as.matrix(sapply(test_set, function(col) {
    if(is.factor(col)) as.numeric(as.character(col)) else as.numeric(col)
  }))
  # Prédiction avec lambda.min (ou "lambda.1se" si on veut plus de régularisation)
  preds <- as.numeric(predict(reg, newx = X_test_mat, s = "lambda.min"))
  return(preds)
}

####################################################
# 4. Sauvegarder 

save(clas, reg, classifieur, regresseur, file = "env.Rdata")

cat("Fichier env.Rdata créé avec succès dans le répertoire courant.\n")
cat("Contenu : clas, reg, classifieur, regresseur\n")
