library(ggplot2)
library(dplyr)

df <- read.csv("data/label_metrics.tsv", sep="\t") %>%
  group_by(label, hier, model) %>%
  summarize(AUC=mean(AUC), AUPR=mean(AUPR), .groups="drop")
df <- df[df$model!="SMC",]

best_med_aucs <- c()
for (level in c("PT", "HLT", "HLGT")) {
  best_med_aucs <- c(best_med_aucs, mean(df[df$model=="CS" & df$hier==level,]$AUC))
  best_med_aucs <- c(best_med_aucs, mean(df[df$model=="RF" & df$hier==level,]$AUC))
}
print(round(range(best_med_aucs), 2))

df2 <-
  df %>%
  group_by(hier, model) %>%
  mutate(outlier = AUC<quantile(AUC,0.25)-1.5*IQR(AUC) | AUC>quantile(AUC,0.75)+1.5*IQR(AUC)) %>%
  ungroup %>%
  filter(outlier==T)

models <- unique(df$model)
models <- levels(reorder(models, models, FUN=function(x) median(df$AUC[df$model==x & df$hier=="PT"])))

g <- ggplot(df, aes(x=factor(model, levels=models), y=AUC, fill=factor(hier, levels=c("PT","HLT","HLGT")))) + 
  stat_boxplot(geom="errorbar") +
  geom_boxplot(outlier.size=0.2, outlier.shape=NA) +
  geom_hline(yintercept=0.5, col="purple") +
  geom_point(data=df2, position=position_jitterdodge(), size=0.3, alpha=0.2) +
  ylim(0,1) +
  xlab("Model") +
  ylab("Side effect AUC") +
  labs(fill="MedDRA level") +
  theme_bw() +
  theme(axis.text.x=element_text(angle=60, vjust=0.6),
        text=element_text(size=10),
        legend.title=element_text(size=6), legend.text=element_text(size=6),
        panel.grid.major=element_blank(), panel.grid.minor=element_blank(),
        legend.direction="horizontal", legend.position=c(.5,.11),
        legend.background=element_rect(color="black"))
ggsave("figures/labelwise_boxplot.pdf", plot=g,
       width=4, height=3)

# Hypothesis testing
levels <- unique(df$hier)
pvals <- c()
for (l in levels) {
  for (m in models) {
    tt <- t.test(subset(df, model==m & hier==l)$AUC, mu=0.5)
    print(paste(l, m, tt$estimate, tt$p.value))
    pvals <- c(pvals, tt$p.value)
  }
}
print(max(pvals, na.rm=T))
