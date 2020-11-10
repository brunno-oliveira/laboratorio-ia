library("caret")
library(ggplot2)

setwd('X://Git//laboratorio-ia//data_source')

df <- read.csv('Material 02 - 8 - R - Admissao - Dados.csv')

# Removendo Serie
df = df[,2:9]

# Separacao de base
set.seed(47)
indices <- createDataPartition(df$ChanceOfAdmit, p=0.80, list=FALSE) 
treino <- df[indices,]
teste <- df[-indices,]

# -----------------------------------------------------------
# RNA
# -----------------------------------------------------------
# -----------------------------------------------------------
# HOLD OUT
set.seed(47)
rna_holdout <- train(
  ChanceOfAdmit~., 
  data=treino, 
  method="nnet",
  linout=T,
  trace=FALSE)
rna_holdout

predicoes.rna_holdout <- predict(rna_holdout, teste)

postResample(pred=predicoes.rna_holdout, obs=teste$ChanceOfAdmit)
cor(predicoes.rna_holdout, teste$ChanceOfAdmit, method = "pearson")

# SYX http://courses.washington.edu/psy315/tutorials/interpretation_of_regression_tutorial.pdf
rna_holdout.r <- cor(predicoes.rna_holdout, teste$ChanceOfAdmit,use = "complete.obs")
sqrt(1 - (rna_holdout.r)^2)

# -------------
# CROSS VALIDATION
set.seed(47)
rna_ctrl <- trainControl(method = "cv", number = 10)
rna_cv <- train(
  ChanceOfAdmit~., 
  data=treino, 
  method="nnet",
  linout=T,
  trace=FALSE, 
  trControl=rna_ctrl)
rna_cv

predict.rna_cv <- predict(rna_cv, teste) 

postResample(pred=predict.rna_cv, obs=teste$ChanceOfAdmit)
cor(predict.rna_cv, teste$ChanceOfAdmit, method = "pearson")

# SYX http://courses.washington.edu/psy315/tutorials/interpretation_of_regression_tutorial.pdf
rna_cv.r <- cor(predict.rna_cv, teste$ChanceOfAdmit, use = "complete.obs")
sqrt(1 - (rna_cv.r)^2)

# -------------
# BEST MODEL
set.seed(47)
rna_grid <- expand.grid(
  size = seq(from = 1, to = 50, by = 5),
  decay = seq(from = 0.1, to = 0.9, by = 0.31)
)
rna_best <- train(
  ChanceOfAdmit~., 
  data = treino, 
  method = "nnet", 
  linout=T,
  tuneGrid = rna_grid, 
  trControl = rna_ctrl, 
  maxit = 2000,
  trace=FALSE) 
rna_best

predict.rna_best <- predict(rna_best, teste) 

postResample(pred=predict.rna_best, obs=teste$ChanceOfAdmit)
cor(predict.rna_best, teste$ChanceOfAdmit, method = "pearson")

# SYX http://courses.washington.edu/psy315/tutorials/interpretation_of_regression_tutorial.pdf
rna_best.r <- cor(predict.rna_best, teste$ChanceOfAdmit, use = "complete.obs")
sqrt(1 - (rna_best.r)^2)


# -----------------------------------------------------------
# KNN
# -----------------------------------------------------------
set.seed(47)
knn_grid <- expand.grid(k = seq(from = 1, to = 20, by = 1))
set.seed(47)
knn <- train(
  ChanceOfAdmit~ ., 
  data=treino,
  method="knn",
  linout=T,
  tuneGrid=knn_grid)
knn

predict.knn <- predict(knn, teste)

postResample(pred=predict.knn, obs=teste$ChanceOfAdmit)
cor(predict.knn, teste$ChanceOfAdmit, method = "pearson")

# SYX http://courses.washington.edu/psy315/tutorials/interpretation_of_regression_tutorial.pdf
knn.r <- cor(predict.knn, teste$ChanceOfAdmit, use = "complete.obs")
sqrt(1 - (knn.r)^2)

# -----------------------------------------------------------
# SVM
# -----------------------------------------------------------
# HOLD OUT
set.seed(47)
svm_holdout <- train(
  ChanceOfAdmit~., 
  data=treino, 
  linout=T,
  method="svmRadial")
svm_holdout

predict.svm_holdout <- predict(svm_holdout, teste) 

postResample(pred=predict.svm_holdout, obs=teste$ChanceOfAdmit)
cor(predict.svm_holdout, teste$ChanceOfAdmit, method = "pearson")

# SYX http://courses.washington.edu/psy315/tutorials/interpretation_of_regression_tutorial.pdf
svm_holdout.r <- cor(predict.svm_holdout, teste$ChanceOfAdmit, use = "complete.obs")
sqrt(1 - (svm_holdout.r)^2)

# -------------
# CROSS VALIDATION
set.seed(47)
svm_ctrl <- trainControl(method = "cv", number = 10)
rna_cv <- train(
  ChanceOfAdmit~., 
  data=treino, 
  method="svmRadial",
  linout=T,
  trace=FALSE, 
  trControl=svm_ctrl)
rna_cv

predict.rna_cv <- predict(rna_cv, teste) 

postResample(pred=predict.rna_cv, obs=teste$ChanceOfAdmit)
cor(predict.rna_cv, teste$ChanceOfAdmit, method = "pearson")

# SYX http://courses.washington.edu/psy315/tutorials/interpretation_of_regression_tutorial.pdf
r <- cor(predict.rna_cv, teste$Volume, use = "complete.obs")
sqrt(1 - (r)^2)

# -------------
# BEST MODEL
svm_grid <- expand.grid(
  sigma = seq(from = 0.1, to = 0.9, by = 0.1)
  ,C = seq(from = 1, to = 5, by = 1))
set.seed(47)
svm_best <- train(
  ChanceOfAdmit~., 
  data=treino, 
  method="svmRadial",
  linout=T,
  trace=FALSE,
  tuneGrid = svm_grid , 
  trControl = svm_ctrl, 
  maxit = 2000)
svm_best

predict.svm_best <- predict(svm_best, teste) 

postResample(pred=predict.svm_best, obs=teste$ChanceOfAdmit)
cor(predict.svm_best, teste$ChanceOfAdmit, method = "pearson")

# SYX http://courses.washington.edu/psy315/tutorials/interpretation_of_regression_tutorial.pdf
svm_best.r <- cor(predict.svm_best, teste$ChanceOfAdmit, use = "complete.obs")
sqrt(1 - (svm_best.r)^2)

# -----------------------------------------------------------
# RF
# -----------------------------------------------------------
# HOLD OUT
set.seed(47)
rf_holdout <- train(
  ChanceOfAdmit~., 
  data=treino,
  linout=T,
  method="rf")
rf_holdout

predict.rf_holdout <- predict(rf_holdout, teste) 

postResample(pred=predict.rf_holdout, obs=teste$ChanceOfAdmit)
cor(predict.rf_holdout, teste$ChanceOfAdmit, method = "pearson")

# SYX http://courses.washington.edu/psy315/tutorials/interpretation_of_regression_tutorial.pdf
rf_holdout.r <- cor(predict.rf_holdout, teste$ChanceOfAdmit, use = "complete.obs")
sqrt(1 - (rf_holdout.r)^2)

# -------------
# CROSS VALIDATION
rf_ctrl <- trainControl(method = "cv", number = 10)
set.seed(47)
rf_cv <- train(
  ChanceOfAdmit~., 
  data=treino, 
  method="rf",
  linout=T,
  trControl=rf_ctrl)
rf_cv

predict.rf_cv <- predict(rf_cv, teste) 

postResample(pred=predict.rf_cv, obs=teste$ChanceOfAdmit)
cor(predict.rf_cv, teste$ChanceOfAdmit, method = "pearson")

# SYX http://courses.washington.edu/psy315/tutorials/interpretation_of_regression_tutorial.pdf
rf_cv.r <- cor(predict.rf_cv, teste$ChanceOfAdmit, use = "complete.obs")
sqrt(1 - (rf_cv.r)^2)

# -------------
# BEST MODEL
set.seed(47)
rf_ctrl <- trainControl(method = "cv", number = 10)
rf_grid = expand.grid(mtry=seq(from = 1, to = 50, by = 1))
rf_best <- train(
  ChanceOfAdmit~., 
  data=treino, 
  method="rf",
  linout=T,
  trControl=rf_ctrl,
  tuneGrid = rf_grid)
rf_best

predict.rf_best <- predict(rf_best, teste) 

postResample(pred=predict.rf_best, obs=teste$ChanceOfAdmit)
cor(predict.rf_best, teste$ChanceOfAdmit, method = "pearson")

# SYX http://courses.washington.edu/psy315/tutorials/interpretation_of_regression_tutorial.pdf
rf_best.r <- cor(predict.rf_best, teste$ChanceOfAdmit, use = "complete.obs")
sqrt(1 - (rf_best.r)^2)

# Grafico de residuos
plot(
  teste$ChanceOfAdmit, 
  teste$ChanceOfAdmit - predict.rf_best, 
  ylab="Residuos", 
  xlab="Change de se admitido", 
  main="RF Best Model Residuos") 
abline(0, 0)  

# Predict New Data
df.new <- read.csv('Material 02 - 8 - R - Admissao - Novos Casos.csv')
df.new$predict <- predict(rf_best, df.new)

