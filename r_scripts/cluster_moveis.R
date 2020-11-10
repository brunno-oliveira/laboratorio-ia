library("klaR")
set.seed(47)

setwd('X://Git//laboratorio-ia//data_source')

df <- read.csv('Material 07 - 2 - Moveis - Dados.csv')

# Quebrando a categoria em N colunas
list_categoria = str_split(df$categoria,'/')
df_categoria = data.frame(matrix(unlist(list_categoria), nrow=length(list_categoria), byrow=T))
colnames(df_categoria) <- c("categoria.1","categoria.2","categoria.3")
df_categoria$categoria.1 = str_trim(df_categoria$categoria.1, side = "both")
df_categoria$categoria.2 = str_trim(df_categoria$categoria.2, side = "both")
df_categoria$categoria.3 = str_trim(df_categoria$categoria.3, side = "both")
df_categoria$categoria.1 = as.factor(df_categoria$categoria.1)
df_categoria$categoria.2 = as.factor(df_categoria$categoria.2)
df_categoria$categoria.3 = as.factor(df_categoria$categoria.3)


# Quebrando a cor em N colunas
list_cor = str_split(df$cor,'/')
df_cor = data.frame(matrix(unlist(list_cor), nrow=length(list_cor), byrow=T))
colnames(df_cor) <- c("cor.1","cor.2")
df_cor$cor.1 = str_trim(df_cor$cor.1, side = "both")
df_cor$cor.2 = str_trim(df_cor$cor.2, side = "both")
df_cor$cor.1 = as.factor(df_cor$cor.1)
df_cor$cor.2 = as.factor(df_cor$cor.2)

# Merge dos 3 dfs
df = cbind(df_categoria, df_cor, df$estilo)

cluster.results <- kmodes(df, 5,iter.max = 20,weighted= FALSE )
cluster.results

df$Cluster = cluster.results[["cluster"]]




