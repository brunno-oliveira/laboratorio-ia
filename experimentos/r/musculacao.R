
install.packages("arules", dep=T)
library(arules)
setwd("/Users/patrick/Studies/iaa_006_007/laboratorio-ia/data_source")
dados <- read.transactions(file="Material 08 – 2 - Musculacao - Dados.csv", format="basket", sep=";")
inspect(dados[1:4])

set.seed(47)
rules <- apriori(dados, parameter = list(supp=0.3, conf=1, target="rules"))
inspect(rules)

