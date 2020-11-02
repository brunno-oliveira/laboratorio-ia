library("caret")
library("e1071")
library("mice")
library("mlbench")
library("kernlab")
library("randomForest")

dados = read.csv("/Users/patrick/Studies/iaa_006_007/laboratorio-ia/data_source/Material 02 - 9 – C - Diabetes - Dados.csv")
#View(dados)

### Limpando a base
dados$num <- NULL
imp <- mice(dados)
dados <- complete(imp, 1)

### Convertendo strings para factor
str(dados)
dados$diabetes <- as.factor(dados$diabetes)
str(dados)

### Particionar a bases em treino (80%) e teste (20%)
set.seed(47)
indices <- createDataPartition(dados$diabetes, p=0.80, list=FALSE) 
treino <- dados[indices,]
teste <- dados[-indices,]

### Treinamento do modelo com o conjunto de treino
set.seed(47)
rna <- train(diabetes~., data=treino, method="nnet",trace=FALSE)
rna

### Predi??es dos valores do conjunto de teste
predicoes.rna <- predict(rna, teste)
confusionMatrix(predicoes.rna, teste$diabetes)

### indica o m?todo cv e numero de folders 10
ctrl <- trainControl(method = "cv", number = 10)
set.seed(47)
rna <- train(diabetes~., data=treino, method="nnet",trace=FALSE, trControl=ctrl)
rna

## Predict
predict.rna <- predict(rna, teste) 
confusionMatrix(predict.rna, teste$diabetes)


### Busca pelo melhor modelo
grid <- expand.grid(size=seq(from=1, to=50, by=5), decay=seq(from=0.1, to=0.9, by=0.3))
set.seed(47)
rna <- train(diabetes~., data=treino, method="nnet", tuneGrid=grid, trControl=ctrl, maxit=2000, trace=FALSE)
rna

# Predict
predict.rna <- predict(rna, teste) 
confusionMatrix(predict.rna, teste$diabetes)




######## SVM ##########
set.seed(47)
svm <- train(diabetes~., data=treino, method="svmRadial")
svm

# Predição SVM
predicoes.svm <- predict(svm, teste)
confusionMatrix(predicoes.svm, teste$diabetes)

# Cross-validation SVM
ctrl <- trainControl(method = "cv", number = 10)
set.seed(47)
svm <- train(diabetes~., data=treino, method="svmRadial", trControl=ctrl)
svm

# Predição SVM com Cross-Validation
predicoes.svm <- predict(svm, teste)
confusionMatrix(predicoes.svm, teste$diabetes)

# Melhor modelo
tuneGrid = expand.grid(C=c(0.1, 0.25, 0.5, 0.75,	1, 2, 3,	5,	8,	13,	21,	34,	50, 55,	89, 100), sigma=c(0.05, 0.1, 0.12, 0.15, 0.2, 0.5, 0.8, 1))
set.seed(47)
svm <- train(diabetes~., data=treino, method="svmRadial", trControl=ctrl, tuneGrid=tuneGrid)
svm

# Predição SVM para o Melhor Modelo
predicoes.svm <- predict(svm, teste)
confusionMatrix(predicoes.svm, teste$diabetes)

######## KNN ##########
tuneGrid = expand.grid(k=c(1,3,5,7,9))
set.seed(47)
knn <- train(diabetes~., data=treino, method="knn", tuneGrid=tuneGrid)
knn

# Predição KNN para o Melhor Modelo
predicoes.knn <- predict(knn, teste)
confusionMatrix(predicoes.knn, teste$diabetes)


######## RandomForest ######
set.seed(47)
rf <- train(diabetes~., data=treino, method="rf")
rf
# Predict
predicoes.rf <- predict(rf, teste)
confusionMatrix(predicoes.rf, teste$diabetes)

# Cross-validation
ctrl <- trainControl(method = "cv", number = 10)
set.seed(47)
rf <- train(diabetes~., data=treino, method="rf", trControl=ctrl)
rf

# Predict
predicoes.rf <- predict(rf, teste)
confusionMatrix(predicoes.rf, teste$diabetes)

# Melhor model
tuneGrid = expand.grid(mtry=c(2, 5, 7, 9))
set.seed(47)
rf <- train(diabetes~., data = treino, method = "rf", trControl=ctrl, tuneGrid = tuneGrid)
rf

# Predict
predicoes.rf <- predict(rf, teste)
confusionMatrix(predicoes.rf, teste$diabetes)
