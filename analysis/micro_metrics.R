library(readr)
library(dplyr)
library(xtable)

df <- read_tsv("data/micro_metrics.tsv")

print_df <- df %>%
  mutate_if(is.numeric, ~(round(., 2)*100)) %>%
  group_by(model) %>%
  summarize(AUC=paste(AUC, collapse="/"),
            AUPR=paste(AUPR, collapse="/"),
            Precision=paste(precision, collapse="/"),
            Recall=paste(recall, collapse="/"))
print(xtable(print_df), include.rownames=F)
