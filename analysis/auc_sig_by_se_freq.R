library(readr)
library(dplyr)
library(ggplot2)

#############
## PROCESS ##
#############

df <- read_tsv("data/label_metrics.tsv") %>%
  group_by(label, hier, model) %>%
  summarize(AUC=mean(AUC), .groups="drop")

plot_df <- data.frame()
for (level in c("PT", "HLT", "HLGT")) {
  y <- read.csv(paste0("../data/SIDER/cid_umls_matrix/MedDRA_",level,".tsv"), sep="\t", row.names=1, check.names=F)
  df_pval_05 <- read.csv(paste0("data/side_effect_05AUC_pvals_",level,".tsv"), sep="\t")
  models <- unique(df_pval_05$Model)
  for (model in models[!models=="SMC"]) {
    df_pval_05_m <- df_pval_05[df_pval_05$Model==model,]
    nfolds_sig <- df_pval_05_m$N_significant_folds
    
    df_ml <- as.data.frame(df[df$model==model & df$hier==level,])
    rownames(df_ml) <- df_ml$label
    auc <- df_ml[as.character(df_pval_05_m$MedDRA_ID),]$AUC
    
    y <- y[as.character(df_pval_05_m$MedDRA_ID)]
    theta <- apply(y, 2, mean)
    
    for (i in 1:length(theta))
      plot_df <- rbind(plot_df, c(level, model, auc[i], theta[i], nfolds_sig[i]))
  }
}
colnames(plot_df) <- c("Level", "Model", "AUC", "theta", "nfolds_sig")
plot_df <- plot_df %>%
  mutate(Level=factor(Level, c("PT","HLT","HLGT")),
         AUC=as.numeric(AUC),
         theta=as.numeric(theta))

##########
## PLOT ##
##########

p <- ggplot(plot_df, aes(x=theta, y=AUC, color=nfolds_sig)) +
  geom_point(size=1, alpha=0.3, shape=16) +
  geom_hline(yintercept=0.5) +
  xlim(c(0,1)) +
  ylim(c(0,1)) +
  xlab("Side effect frequency") +
  ylab("Side effect AUC") +
  facet_grid(Model~Level) +
  labs(color="Number of significant folds") +
  guides(color=guide_legend(nrow=1, override.aes=list(size=3))) +
  theme_bw() +
  theme(strip.placement="outside", strip.background=element_blank(),
        legend.position="bottom",
        panel.grid.major=element_blank(), panel.grid.minor=element_blank())
ggsave("figures/AUC_by_SEfreq_wSig.pdf", p, width=7, height=10)

p <- ggplot(subset(plot_df, Model=="RF" & Level=="HLGT"),
            aes(x=theta, y=AUC, color=nfolds_sig)) +
  geom_point(size=1, alpha=0.5, shape=16) +
  geom_hline(yintercept=0.5) +
  xlim(c(0,1)) +
  ylim(c(0,1)) +
  xlab("Side effect frequency") +
  ylab("Side effect AUC") +
  labs(color="Number of significant folds") +
  guides(color=guide_legend(nrow=2, byrow=T, override.aes=list(size=3))) +
  theme_bw() +
  theme(strip.placement="outside", strip.background=element_blank(),
        legend.position=c(0.6,0.15), legend.direction="horizontal",
        panel.grid.major=element_blank(), panel.grid.minor=element_blank())
ggsave("figures/AUC_by_SEfreq_wSig_RF_HLGT.pdf", p, width=5, height=4)
