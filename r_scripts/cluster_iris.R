library(tidyverse)
data("iris")
df = iris

# Transformando SPECIES em OneHot
df$setora <- as.factor(ifelse(df$Species=="setosa", 1, 0))
df$versicolor <- as.factor(ifelse(df$Species=="versicolor", 1, 0))
df$Species <- NULL

set.seed(47)
fviz_nbclust(df, kmeans, method = "wss")

k <- kmeans(df, centers = 2, nstart = 25)
df$Cluster = k[["cluster"]]
