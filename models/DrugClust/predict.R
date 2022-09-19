library("DrugClust")
library("cclust")

args <- commandArgs(trailingOnly=T)
id <- args[1]

X <- as.matrix(read.delim(paste0("temp/",id,"-predict_X.csv"), row.names=1))
load(paste0("temp/",id,"-model"))

PredictionKMeans<-function(A,cl,test,nlabels){
  predizioni=matrix(nrow=nrow(test), ncol=nlabels)
  cl$centers[is.nan(cl$centers)]<-0
  ycl<-predict(cl,test,type="both")
  for(i in 1:nrow(test)){
    num_cluster<-ycl$cluster[i]
    vettoreA<-A[num_cluster,]
    predizioni[i,]<-vettoreA
  }
  return(predizioni)
}
y <- PredictionKMeans(A, cl, X, length(label_names))

write.table(y, paste0("temp/",id,"-predict_y.csv"),
            row.names=rownames(X), col.names=label_names, sep="\t")
