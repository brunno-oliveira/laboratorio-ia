library("caret")

setwd('X://Git//laboratorio-ia//data_source')

df <- read.csv('Material 02 - 4 - R - Biomassa - Dados.csv')

# Separacao de base
indices <- createDataPartition(df$biomassa, p=0.80, list=FALSE) 
treino <- df[indices,]
teste <- df[-indices,]

# -----------------------------------------------------------
# RNA
# -----------------------------------------------------------
# -----------------------------------------------------------
# HOLD OUT
set.seed(47)
rna_holdout <- train(biomassa~., data=treino, method="nnet",trace=FALSE)
rna_holdout

predicoes.rna_holdout <- predict(rna_holdout, teste)

postResample(pred=predicoes.rna_holdout, obs=teste$biomassa)