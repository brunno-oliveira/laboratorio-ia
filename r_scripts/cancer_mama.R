library("caret")
library(pROC)
library(ROCR)
set.seed(47)

setwd('X://Git//laboratorio-ia//data_source')


df <- read.csv('Material 02 - 2 - Cancer de Mama - Dados.csv')

# Removendo o ID do dataset
df = df[,2:11]

# Separacao de base
indices <- createDataPartition(df$Class, p=0.80, list=FALSE) 
treino <- df[indices,]
teste <- df[-indices,]

# -----------------------------------------------------------
# RNA
# -----------------------------------------------------------

# HOLD OUT
set.seed(47)
rna_holdout <- train(Class~., data=treino, method="nnet",trace=FALSE)
rna_holdout

predicoes.rna_holdout <- predict(rna_holdout, teste)
confusionMatrix(predicoes.rna_holdout, teste$Class)

# ROC
set.seed(47)
predict.proba.rna_holdout <- predict(rna_holdout, teste, type='prob')
predict.proba.rna_holdout = predict.proba.rna_holdout$benign

boolClass <- ifelse(teste$Class=="benign", 1, 0)

# Buscando o melhor corte com base no RECALL para penalisar o FALSO NEGATIVO
set.seed(47)
prediction.proba.rna_holdout <- prediction(predict.proba.rna_holdout, boolClass)
eval.proba.rna_holdout = performance(prediction.proba.rna_holdout, "rec")
plot(eval.proba.rna_holdout)

max <- which.max(slot(eval.proba.rna_holdout, "y.values")[[1]])
rec <- slot(eval.proba.rna_holdout, "y.values")[[1]][max]
cut <- slot(eval.proba.rna_holdout, "x.values")[[1]][max]
abline(h=rec,v=cut,col="red")

# ROC CURVE
eval.proba.rna_holdout = performance(prediction.proba.rna_holdout, "tpr", "fpr")
plot(eval.proba.rna_holdout, colorize=T)
abline(a=0,b=1,)

# AUC
auc <- performance(prediction.proba.rna_holdout, "auc")
auc <- round(unlist(slot(auc, "y.values")),4)
auc

# CROSS VALIDATION
rna_ctrl <- trainControl(method = "cv", number = 10)
rna_cv <- train(
  Class~., 
  data=treino, 
  method="nnet",
  trace=FALSE, trControl=rna_ctrl)
rna_cv

predict.rna_cv <- predict(rna_cv, teste) 
confusionMatrix(predict.rna_cv, as.factor(teste$Class))

# BEST MODEL
rna_grid <- expand.grid(
  size = seq(from = 1, to = 50, by = 5),
  decay = seq(from = 0.1, to = 0.9, by = 0.31)
)
rna_best <- train(
  form = Class~. , 
  data = treino , 
  method = "nnet" , 
  tuneGrid = rna_grid , 
  trControl = rna_ctrl , 
  maxit = 2000,
  trace=FALSE) 
rna_best

predict.rna_best <- predict(rna_best, teste) 
confusionMatrix(predict.rna_best, as.factor(teste$Class))

# -----------------------------------------------------------
# SVM
# -----------------------------------------------------------

# HOLD OUT
set.seed(47)
svm_holdout <- train(
  Class~., 
  data=treino, 
  method="svmRadial",
  trace=FALSE)
svm_holdout

predict.svm_holdout <- predict(svm_holdout, teste) 
confusionMatrix(predict.svm_holdout, as.factor(teste$Class))

# CROSS VALIDATION
svm_ctrl <- trainControl(method = "cv", number = 10)
set.seed(47)
svm_cv <- train(
  Class~., 
  data=treino, 
  method="svmRadial",
  trControl=svm_ctrl,
  trace=FALSE)
svm_cv

predict.svm_cv <- predict(svm_cv, teste) 
confusionMatrix(predict.svm_cv, as.factor(teste$Class))

# BEST MODEL
svm_grid <- expand.grid(
  sigma = seq(from = 0.1, to = 0.9, by = 0.1)
  ,C = seq(from = 1, to = 5, by = 1))
set.seed(47)
svm_best <- train(
  Class~., 
  data=treino, 
  method="svmRadial",
  trace=FALSE,
  tuneGrid = svm_grid , 
  trControl = svm_ctrl, 
  maxit = 2000)
svm_best

predict.svm_best <- predict(svm_best, teste) 
confusionMatrix(predict.svm_best, as.factor(teste$Class))

# -----------------------------------------------------------
# KNN
# -----------------------------------------------------------
knn_grid <- expand.grid(k = seq(from = 1, to = 20, by = 1))
set.seed(47)
knn <- train(
  Class~ ., 
  data=treino,
  method="knn",
  tuneGrid=knn_grid)
knn

predict.knn <- predict(knn, teste) 
confusionMatrix(predict.knn, as.factor(teste$Class))

# -----------------------------------------------------------
# RF
# -----------------------------------------------------------
# HOLD OUT
set.seed(47)
rf_holdout <- train(
  Class~., 
  data=treino, 
  method="rf")
rf_holdout

predict.rf_holdout <- predict(rf_holdout, teste) 
confusionMatrix(predict.rf_holdout, as.factor(teste$Class))

# CROSS VALIDATION
rf_ctrl <- trainControl(method = "cv", number = 10)
set.seed(47)
rf_cv <- train(
  Class~., 
  data=treino, 
  method="rf",
  trControl=rf_ctrl)
rf_cv

predict.rf_cv <- predict(rf_cv, teste) 
confusionMatrix(predict.rf_cv, as.factor(teste$Class))

# BEST MODEL
set.seed(47)
rf_grid = expand.grid(mtry=seq(from = 1, to = 10, by = 1))
rf_best <- train(
  Class~., 
  data=treino, 
  method="rf",
  trControl=rf_ctrl,
  tuneGrid = rf_grid)
rf_best

predict.rf_best <- predict(rf_best, teste) 
confusionMatrix(predict.rf_best, as.factor(teste$Class))








