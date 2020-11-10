#library("caret")
library(tidyverse)
library(cluster)  
library(factoextra)
set.seed(47)

setwd('X://Git//laboratorio-ia//data_source')


df <- read.csv('Material 02 - 5 - C - Veiculos - Dados.csv')

# Removendo a coluna 'a' do dataset
df = df[,2:20]

# Transformando TIPO do veiculo em OneHot
df$van <- as.factor(ifelse(df$tipo=="van", 1, 0))
df$saab <- as.factor(ifelse(df$tipo=="saab", 1, 0))
df$bus <- as.factor(ifelse(df$tipo=="bus", 1, 0))
df$tipo <- NULL

set.seed(47)
fviz_nbclust(df, kmeans, method = "silhouette")

set.seed(47)
k <- kmeans(df, centers = 2, nstart = 25)
df$Cluster = k[["cluster"]]


