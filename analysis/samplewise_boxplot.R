library(ggplot2)
library(dplyr)

df <- read.csv("data/sample_metrics.tsv", sep="\t")
df2 <- df %>%
  group_by(hier, model) %>%
  mutate(outlier = AUC < quantile(AUC,0.25)-1.5*IQR(AUC) | AUC > quantile(AUC,0.75)+1.5*IQR(AUC)) %>%
  ungroup %>%
  filter(outlier==T)

models <- unique(df$model)
models <- levels(reorder(models, models, FUN=function(x) median(df$AUC[df$model==x & df$hier=="PT"])))
models <- c("SMC", models[models != "SMC"])

g <- ggplot(df, aes(x=factor(hier, levels=c("PT","HLT","HLGT")), y=AUC, fill=factor(model, levels=models))) + 
  stat_boxplot(geom="errorbar") +
  geom_boxplot(outlier.size=0.2, outlier.shape=NA) +
  geom_point(data=df2, position=position_jitterdodge(), size=0.3, alpha=0.2) +
  ylim(0,1) +
  xlab("MedDRA level") +
  ylab("Drug AUC") +
  labs(fill="Model") +
  theme_bw() +
  theme(text=element_text(size=13),
        legend.title=element_text(size=12), legend.text=element_text(size=10),
        panel.grid.major=element_blank(), panel.grid.minor=element_blank(),
        legend.direction="vertical", legend.position="right")
ggsave("figures/samplewise_boxplot.pdf", plot=g,
       width=6, height=3)

# Hypothesis testing
levels <- unique(df$hier)
pvals <- c()
for (l in levels) {
  for (m in models) {
    tt <- t.test(subset(df, model=="SMC" & hier==l)$AUC,
                 subset(df, model==m & hier==l)$AUC,
                 paired=T)
    print(paste(l, m, tt$estimate, tt$p.value))
    pvals <- c(pvals, tt$p.value)
  }
}
print(max(pvals, na.rm=T))
