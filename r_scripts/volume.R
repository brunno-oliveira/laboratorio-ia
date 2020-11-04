library("caret")
library("ggpubr")

setwd('X://Git//laboratorio-ia//data_source')

df <- read.csv('Material 02 - 3 - Estimativa de Volume - Dados.csv')

# Separacao de base
indices <- createDataPartition(df$Volume, p=0.80, list=FALSE) 
treino <- df[indices,]
teste <- df[-indices,]

# -----------------------------------------------------------
# RNA
# -----------------------------------------------------------
# -----------------------------------------------------------
# HOLD OUT
set.seed(47)
rna_holdout <- train(Volume~., data=treino, method="nnet", linout=T, trace=FALSE)
rna_holdout

predicoes.rna_holdout <- predict(rna_holdout, teste)

postResample(pred=predicoes.rna_holdout, obs=teste$Volume)

cor(predicoes.rna_holdout, teste$Volume, method = "pearson")

# -----------------------------------------------------------
# CROSS VALIDATION
set.seed(47)
rna_ctrl <- trainControl(method = "cv", number = 10)
rna_cv <- train(
  Volume~., 
  data=treino, 
  method="nnet",
  linout=T,
  trace=FALSE, 
  trControl=rna_ctrl)
rna_cv

predict.rna_cv <- predict(rna_cv, teste) 

postResample(pred=predict.rna_cv, obs=teste$Volume)

cor(predict.rna_cv, teste$Volume, method = "pearson")

# -----------------------------------------------------------
# BEST MODEL
set.seed(47)
rna_grid <- expand.grid(
  size = seq(from = 1, to = 50, by = 5),
  decay = seq(from = 0.1, to = 0.9, by = 0.31)
)
rna_best <- train(
  Volume~., 
  data = treino, 
  method = "nnet", 
  linout=T,
  tuneGrid = rna_grid, 
  trControl = rna_ctrl, 
  maxit = 2000,
  trace=FALSE) 
rna_best

predict.rna_best <- predict(rna_best, teste) 

postResample(pred=predict.rna_best, obs=teste$Volume)

cor(predict.rna_best, teste$Volume, method = "pearson")

# -----------------------------------------------------------
# KNN
# -----------------------------------------------------------
set.seed(47)
knn_grid <- expand.grid(k = seq(from = 1, to = 20, by = 1))
set.seed(47)
knn <- train(
  Volume~ ., 
  data=treino,
  method="knn",
  tuneGrid=knn_grid)
knn

predict.knn <- predict(knn, teste) 

