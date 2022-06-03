library(EnvStats)
library(ggplot2)
library(gridExtra)
library(xtable)

stats_df <- data.frame(n=c(NA,NA,NA), mu=NA, sigma=NA,
                       ab=NA, mu_beta=NA, sigma_beta=NA, auc=NA,
                       row.names=c("PT","HLT","HLGT"))

plot_level <- function(level, bins, x_cutoff=1, ylab="", N=10^6, legend.position="None") {
  set.seed(42)
  
  Y <- read.csv(paste0("../data/SIDER/cid_umls_matrix/MedDRA_", level, ".tsv"), sep="\t", row.names=1, check.names=F)
  theta <- apply(Y, MARGIN=2, mean)
  stats_df[level, "n"] <<- length(theta)
  stats_df[level, "mu"] <<- mean(theta)
  stats_df[level, "sigma"] <<- sd(theta)
  
  fit <- ebeta(theta, method="mle")
  alpha <- as.numeric(fit$parameters[1])
  beta <- as.numeric(fit$parameters[2])
  stats_df[level, "ab"] <<- paste0(round(alpha,2), ", ", round(beta,2))
  stats_df[level, "mu_beta"] <<- alpha/(alpha+beta)
  stats_df[level, "sigma_beta"] <<- sqrt( (alpha*beta)/((alpha+beta)^2 * (alpha+beta+1)) )
  stats_df[level, "auc"] <<- mean(rbeta(N, alpha+1, beta) > rbeta(N, alpha, beta+1))
  
  B <- bins
  h <- c()
  for (i in seq(0,x_cutoff,x_cutoff/B)[1:B]) {
    h <- c(h, (pbeta(i+x_cutoff/B, alpha, beta) - pbeta(i, alpha, beta)) * length(theta)) 
  }
  hi <- hist(theta[theta<=x_cutoff], breaks=seq(0,x_cutoff,x_cutoff/B))$counts
  if (x_cutoff < 1) {
    h <- c(h, (pbeta(1, alpha, beta) - pbeta(x_cutoff, alpha, beta)) * length(theta))
    hi <- c(hi, sum(theta>x_cutoff))
  }
  plot_df <- data.frame(h=h, hi=hi)
  
  xs <- seq(0,x_cutoff,x_cutoff/B)
  if (x_cutoff < 1)
    xsa <- xs
  else
    xsa <- xs[1:B]
  g <- ggplot(plot_df, aes(x=xsa)) +
    geom_bar(aes(y=hi, fill="data"), stat="identity", position=position_nudge((x=1/2 * x_cutoff/B)), width=x_cutoff/B, alpha=1) +
    geom_bar(aes(y=h, fill="theory"), stat="identity", position=position_nudge((x=1/2 *x_cutoff/B)), width=x_cutoff/B, alpha=0.6) +
    xlab(expression(theta[i])) +
    ylab(ylab) +
    scale_fill_manual(name=NA, values=c("data"="blue", "theory"="red"),
                      labels=c("Empirical", "Beta fit")) +
    geom_text(x=0.35*x_cutoff, y=0.95*max(max(h), max(hi)),
              label=level, size=3, check_overlap=T) +
    geom_text(x=0.35*x_cutoff, y=0.89*max(max(h), max(hi)),
              label=paste0("italic(n)==",length(theta)), parse=T, size=3, check_overlap=T) +
    theme_bw() +
    theme(panel.grid.major=element_blank(), panel.grid.minor=element_blank(),
          legend.title=element_blank(), legend.position=legend.position,
          legend.text=element_text(size=8), legend.key.size = unit(0.3, "cm"))
  g
}

g1 <- plot_level("PT", 50, 0.2, "Frequency")
g2 <- plot_level("HLT", 50, 0.45)
g3 <- plot_level("HLGT", 50, 1, legend.position=c(0.8,0.9))
pdf("figures/SIDER_freq_distrs.pdf", width=9, height=3)
grid.arrange(g1,g2,g3, nrow=1)
dev.off()

print(xtable(stats_df, digits=3))
