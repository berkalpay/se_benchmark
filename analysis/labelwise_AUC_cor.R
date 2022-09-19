library(tidyr)
library(dplyr)
library(corrplot)

df <- read.csv("data/label_metrics.tsv", sep="\t") %>%
  group_by(label, hier, model) %>%
  summarise(AUC=mean(AUC), .groups="drop")

for (level in c("PT", "HLT", "HLGT")) {
  auc_by_label <- spread(df[df$hier==level,c("label","model","AUC")], key=model, value=AUC)
  auc_by_label <- auc_by_label[,c("DrugClust","kNN","SVM","CS","MLKNN","RF")]
  
  pdf(paste0("figures/labelwise_AUC_cor_", level, ".pdf"), height=4.2, width=4.2)
  corrplot(cor(auc_by_label), title=level,
           type="lower", order="original", tl.col="black", addCoef.col="black",
           method="shade", diag=F, cl.pos="n",
           mar=c(0,0,2,0))
  dev.off()
}
