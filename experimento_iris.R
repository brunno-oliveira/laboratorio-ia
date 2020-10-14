### Pacotes necessários:
library("caret")

### Leitura dos dados
data(iris)
dados <- iris 
View(dados)

### Particionar a bases em treino (80%) e teste (20%)
set.seed(1912)
indices <- createDataPartition(dados$Species, p=0.80, list=FALSE) 
treino <- dados[indices,]
teste <- dados[-indices,]

### Treinamento do modelo com o conjunto de treino
set.seed(1912)
rna <- train(Species~., data=treino, method="nnet",trace=FALSE)
rna

### Predições dos valores do conjunto de teste
predicoes.rna <- predict(rna, teste)

### Matriz de confusão
confusionMatrix(predicoes.rna, teste$Species)

### indica o método cv e numero de folders 10
ctrl <- trainControl(method = "cv", number = 10)

### executa a RNA com esse ctrl
rna <- train(Species~., data=treino, method="nnet",trace=FALSE, trControl=ctrl)
predict.rna <- predict(rna, teste) 
confusionMatrix(predict.rna, as.factor(teste$Species))



