
install.packages("arules", dep=T)
library(arules)
setwd("/Users/patrick/Studies/iaa_006_007/laboratorio-ia/data_source")
dados <- read.transactions(file="Material 08 – 1 - Lista de Compras - Dados.csv", format="basket", sep=";")
inspect(dados[1:4])

set.seed(47)
rules <- apriori(dados, parameter = list(supp=0.1, conf=0.8, target="rules"))
inspect(rules)

