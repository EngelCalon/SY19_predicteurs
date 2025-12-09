df.reg = read.table("data/TP5_a25_reg_app.txt")
df.class = read.table("data/TP5_a25_clas_app.txt")

summary(df.reg)
summary(df.class)

folds = 10

df.reg.random <- df.reg[sample(nrow(df.reg)),]

for (k1 in 1:K){
  m = length(df.reg.random)
  selected <- df.reg.random[k1*m : (k1+1)*m, ]
}