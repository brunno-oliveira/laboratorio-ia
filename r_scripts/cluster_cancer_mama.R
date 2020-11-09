#library("caret")
library(tidyverse)
library(cluster)  
library(factoextra)
set.seed(47)

setwd('X://Git//laboratorio-ia//data_source')


df <- read.csv('Material 02 - 2 - Cancer de Mama - Dados.csv')

# Removendo o ID do dataset
df = df[,2:11]
df$Class <- ifelse(df$Class=="benign", 1, 0)

set.seed(47)
fviz_nbclust(df, kmeans, method = "wss")

k <- kmeans(df, centers = 2, nstart = 25)
df$Cluster = k[["cluster"]]


