library("caret")
library("e1071")
library("mice")
library("mlbench")

dados = read.csv("/Users/patrick/Studies/iaa_006_007/laboratorio-ia/data_source/Material 02 - 6 - C - Previsao do Tempo - Dados.csv")
# View(dados)

### Convertendo strings para factor
str(dados)
dados$Ceu <- as.factor(dados$Ceu)
dados$Temperatura <- as.factor(dados$Temperatura)
dados$Umidade <- as.factor(dados$Umidade)
dados$Vento <- as.factor(dados$Vento)
dados$Chovera <- as.factor(dados$Chovera)
str(dados)

### Particionar a bases em treino (80%) e teste (20%)
set.seed(47)
indices <- createDataPartition(dados$Chovera, p=0.80, list=FALSE) 
treino <- dados[indices,]
teste <- dados[-indices,]

### Treinamento do modelo com o conjunto de treino
set.seed(47)
rna <- train(Chovera~., data=treino, method="nnet",trace=FALSE)
rna

### Predi??es dos valores do conjunto de teste
predicoes.rna <- predict(rna, teste)
confusionMatrix(predicoes.rna, teste$Chovera)

### indica o m?todo cv e numero de folders 10
ctrl <- trainControl(method = "cv", number = 10)
set.seed(47)
rna <- train(Chovera~., data=treino, method="nnet",trace=FALSE, trControl=ctrl)
rna

## Predict
predict.rna <- predict(rna, teste) 
confusionMatrix(predict.rna, teste$Chovera)


### Busca pelo melhor modelo
grid <- expand.grid(size=seq(from=1, to=45, by=10), decay=seq(from=0.1, to=0.9, by=0.3))
set.seed(47)
rna <- train(Chovera~., data=treino, method="nnet", tuneGrid=grid, trControl=ctrl, maxit=2000, trace=FALSE)
rna

# Predict
predict.rna <- predict(rna, teste) 
confusionMatrix(predict.rna, teste$Chovera)



######## SVM ##########
set.seed(47)
svm <- train(Chovera~., data=treino, method="svmRadial")
svm

# Predição SVM
predicoes.svm <- predict(svm, teste)
confusionMatrix(predicoes.svm, teste$Chovera)

# Cross-validation SVM
ctrl <- trainControl(method = "cv", number = 10)
set.seed(47)
svm <- train(Chovera~., data=treino, method="svmRadial", trControl=ctrl)
svm

# Predição SVM com Cross-Validation
predicoes.svm <- predict(svm, teste)
confusionMatrix(predicoes.svm, teste$Chovera)

# Melhor modelo
tuneGrid = expand.grid(C=c(1,2,10,50,100), sigma=c(0.1, 0.15, 0.2))

set.seed(47)
svm <- train(Chovera~., data=treino, method="svmRadial", trControl=ctrl, tuneGrid=tuneGrid)
svm

# Predição SVM para o Melhor Modelo
predicoes.svm <- predict(svm, teste)
confusionMatrix(predicoes.svm, teste$Chovera)



######## KNN ##########
tuneGrid = expand.grid(k=c(1,3,5,7,9))
set.seed(47)
knn <- train(Chovera~., data=treino, method="knn", tuneGrid=tuneGrid)
knn

# Predição KNN para o Melhor Modelo
predicoes.knn <- predict(knn, teste)
confusionMatrix(predicoes.knn, teste$Chovera)


######## RandomForest ######
set.seed(47)
rf <- train(Chovera~., data=treino, method="rf")
rf
# Predict
predicoes.rf <- predict(rf, teste)
confusionMatrix(predicoes.rf, teste$Chovera)

# Cross-validation
ctrl <- trainControl(method = "cv", number = 10)
set.seed(47)
rf <- train(Chovera~., data=treino, method="rf", trControl=ctrl)
rf

# Predict
predicoes.rf <- predict(rf, teste)
confusionMatrix(predicoes.rf, teste$Chovera)

# Melhor model
tuneGrid = expand.grid(mtry=c(2, 5, 7, 9))
set.seed(47)
rf <- train(Chovera~., data = treino, method = "rf", trControl=ctrl, tuneGrid = tuneGrid)
rf

# Predict
predicoes.rf <- predict(rf, teste)
confusionMatrix(predicoes.rf, teste$Chovera)