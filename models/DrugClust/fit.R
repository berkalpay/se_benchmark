library("DrugClust")

args <- commandArgs(trailingOnly=T)
id <- args[1]
num_clusters <- as.integer(args[2])

X <- as.matrix(read.delim(paste0("temp/",id,"-fit_X.csv"), header=T, row.names=1))
y <- as.matrix(read.delim(paste0("temp/",id,"-fit_y.csv"), header=T, row.names=1))
label_names <- colnames(y)

cl <- KMeans(X, num_clusters) # cclust:cclust(train,num_clusters,verbose=TRUE,method="kmeans")
KMeansModel<-function(train,trainpharmat,num_clusters,cl){
  A <- matrix(nrow=num_clusters,ncol=ncol(trainpharmat)) #matrice ADIJ
  v<-numeric(ncol(trainpharmat))
  for(n in 1:num_clusters){
    k<-which(cl$cluster==n)
    if(length(k)>1){
      drugs_cluster<-trainpharmat[k,]
      SommaSideEffect<-colSums(drugs_cluster)
      SommaTotSE<-colSums(trainpharmat)
      NumeroDrugsCluster=length(k)
      for(y in 1:ncol(trainpharmat)){
        P1=as.numeric(SommaSideEffect[y])/(SommaTotSE[y])
        P2=as.numeric(SommaTotSE[y])/nrow(train)
        P3=NumeroDrugsCluster/nrow(train)
        A[n,y]<-P1*P2/P3
      }
    }
    else{
      A[n,]<-v
    }
  }
  return(A)
}
A <- KMeansModel(X, y, num_clusters, cl)

save(A, cl, label_names, file=paste0("temp/",id,"-model"))
