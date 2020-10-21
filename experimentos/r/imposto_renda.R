library("caret")
library("e1071")
library("mice")
library("mlbench")

dados = read.csv("/Users/patrick/Studies/iaa_006_007/laboratorio-ia/data_source/Material 02 - 7 – C - IR - Dados.csv")
View(dados)

### Convertendo strings para factor
str(dados)
dados$rest <- as.factor(dados$rest)
dados$ecivil <- as.factor(dados$ecivil)
dados$sonegador <- as.factor(dados$sonegador)
str(dados)

### Particionar a bases em treino (80%) e teste (20%)
set.seed(47)
indices <- createDataPartition(dados$sonegador, p=0.80, list=FALSE) 
treino <- dados[indices,]
teste <- dados[-indices,]

### Treinamento do modelo com o conjunto de treino
rna <- train(sonegador~., data=treino, method="nnet",trace=FALSE)
rna

### Predi??es dos valores do conjunto de teste
predicoes.rna <- predict(rna, teste)
confusionMatrix(predicoes.rna, teste$sonegador)

### indica o m?todo cv e numero de folders 10
ctrl <- trainControl(method = "cv", number = 10)
rna <- train(sonegador~., data=treino, method="nnet",trace=FALSE, trControl=ctrl)
rna

## Predict
predict.rna <- predict(rna, teste) 
confusionMatrix(predict.rna, teste$sonegador)


### Busca pelo melhor modelo
grid <- expand.grid(size=seq(from=1, to=45, by=10), decay=seq(from=0.1, to=0.9, by=0.3))
set.seed(47)
rna <- train(sonegador~., data=treino, method="nnet", tuneGrid=grid, trControl=ctrl, maxit=2000, trace=FALSE)
rna

# Predict
predict.rna <- predict(rna, teste) 
confusionMatrix(predict.rna, teste$sonegador)
