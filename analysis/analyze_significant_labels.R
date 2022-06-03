library(readr)
library(tidyr)
library(dplyr)
library(UpSetR)
library(grid)

df <- read_tsv("data/label_metrics.tsv")

# Count number of folds each side effect is significant in
df_nsig <- df %>%
  group_by(label, model, hier) %>%
  summarize(nfolds_sig=sum(AUC05_pval<0.05))

# Upset plots
df_sig <- df_nsig[df_nsig$nfolds_sig==5,]

for (level in c("PT", "HLT", "HLGT")) {
  df_sig_level <- df_sig[df_sig$hier==level, c("label", "model")]
  df_sig_level_bin <- data.frame(unclass(table(df_sig_level)))
  
  n_se <- length(unique(df[df$hier==level,]$label))
  print(level)
  print(paste("RF:", round(sum(df_sig_level$model == "RF")/n_se, 3)))
  print(paste("CS:", round(sum(df_sig_level$model == "CS")/n_se, 3)))
  
  pdf(paste0("figures/sig_upset_", level, ".pdf"), onefile=F,
      width=8, height=3)
  print(upset(df_sig_level_bin,
              nsets=length(unique(df_sig_level$model)),
              nintersects=NA,
              set_size.scale_max=n_se))
  grid.text(level, y=0.97)
  dev.off()
}

# Output side effect names and how many models they were significant in
umls_df <- read.csv("../data/UMLS/MRCONSO.RRF", sep="|", header=F)[c(11,13,15)]
umls_df[is.na(umls_df)] <- "empty"

for (level in c("PT", "HLT", "HLGT")) {
  if (level == "PT") {
    meddra_level_code <- "PT"
  } else if (level == "HLT") {
    meddra_level_code <- "HT"
  } else if (level == "HLGT") {
    meddra_level_code <- "HG"
  }
  umls_dfl <- umls_df[umls_df$V13==meddra_level_code,]
  
  meddra_name <- function(label) umls_dfl[umls_dfl$V11 == label,]$V15
  df_level <- df[df$hier==level, c("fold", "label", "model", "AUC05_pval")]
  df_pval_05 <- df_level %>%
    pivot_wider(names_from=fold, values_from=AUC05_pval, names_prefix="Fold")
  df_pval_05$name <- sapply(df_pval_05$label, meddra_name)
  df_pval_05 <- df_pval_05[c(1, ncol(df_pval_05), seq(2,ncol(df_pval_05)-1))]
  
  df_pval_05$nfolds_sig <- apply(df_pval_05[4:ncol(df_pval_05)], 1, function(v) sum(v<0.05))
  
  colnames(df_pval_05)[c(1,2,3,ncol(df_pval_05))] <- c("MedDRA_ID", "Description",
                                                       "Model", "N_significant_folds")
  write_tsv(df_pval_05, paste0("data/side_effect_05AUC_pvals_",level,".tsv"))
}
