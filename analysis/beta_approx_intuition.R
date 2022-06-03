N <- 10^6

pdf("../analysis/figures/beta_approx_intuition.pdf", height=4, width=8)
par(mar=c(2,2,1,1), oma = c(2.1,2.1,0,0), mfrow=c(1,2))

# variance "same" (0.0175), mean different (0.5, 0.9)
aa1 <- 6.642857142857142
ba1 <- 6.642857142857142
aa2 <- 3.7285714285714273
ba2 <- 0.41428571428571404
dbeta_plot1 <- function(x) dbeta(x, aa1, ba1+1)
curve(dbeta_plot1, col="red", lty="dotted",
      yaxt="n")
axis(side=2, at=c(0,1,2,3))
dbeta_plot2 <- function(x) dbeta(x, aa1+1, ba1)
curve(dbeta_plot2, add=T, col="red", lty="dashed")
dbeta_plot3 <- function(x) dbeta(x, aa2, ba2+1)
curve(dbeta_plot3, add=T, col="blue", lty="dotted")
dbeta_plot4 <- function(x) dbeta(x, aa2+1, ba2)
curve(dbeta_plot4, add=T, col="blue", lty="dashed")

legend("topleft", legend=c(expression(paste("X~Beta(",alpha+1,", ",beta,")")),
                           expression(paste("Y~Beta(",alpha, ", ", beta+1, ")"))),
       lty=c("dashed","dotted"), cex=0.65, bty="n")
text(0.15, y=2, labels=c(expression(paste(alpha=="6.64, ", beta=="6.64"))), col="red", cex=0.65)
text(0.85, y=2.7, labels=c(expression(paste(alpha=="3.72, ", beta=="0.41"))), col="blue", cex=0.65)

pa1 <- mean(rbeta(N, aa1+1,ba1) > rbeta(N,aa1,ba1+1))
text(0.1, y=1.8, labels=c(paste0("p(X>Y)=", round(pa1,2))), col="red", cex=0.6)
pa2 <- mean(rbeta(N, aa2+1,ba2) > rbeta(N,aa2,ba2+1))
text(0.8, y=2.5, labels=c(paste0("p(X>Y)=", round(pa2,2))), col="blue", cex=0.6)

# mean "same" (0.25, 0.75), variance different (.01, .03)
ab1 <- 4.4375
bb1 <- 13.3125
ab2 <- 3.938
bb2 <- 1.313
dbeta_plot1 <- function(x) dbeta(x, ab1, bb1+1)
curve(dbeta_plot1, col="red", lty="dotted",
      xlab="", ylab="")
dbeta_plot2 <- function(x) dbeta(x, ab1+1, bb1)
curve(dbeta_plot2, add=T, col="red", lty="dashed")
dbeta_plot3 <- function(x) dbeta(x, ab2, bb2+1)
curve(dbeta_plot3, add=T, col="blue", lty="dotted")
dbeta_plot4 <- function(x) dbeta(x, ab2+1, bb2)
curve(dbeta_plot4, add=T, col="blue", lty="dashed")

text(0.15, y=4, labels=c(expression(paste(alpha=="4.44, ", beta=="13.31"))), col="red", cex=0.65)
text(0.85, y=3.7, labels=c(expression(paste(alpha=="3.94, ", beta=="1.31"))), col="blue", cex=0.65)

pb1 <- mean(rbeta(N, ab1+1,bb1) > rbeta(N,ab1,bb1+1))
text(0.1, y=3.7, labels=c(paste0("p(X>Y)=", round(pb1,2))), col="red", cex=0.6)
pb2 <- mean(rbeta(N, ab2+1,bb2) > rbeta(N,ab2,bb2+1))
text(0.82, y=3.4, labels=c(paste0("p(X>Y)=", round(pb2,2))), col="blue", cex=0.6)

title(xlab=expression("x"), ylab="f(x)", outer=T, line=1)
dev.off()

print(mean(rbeta(N, aa1+1,ba1) > rbeta(N,aa1,ba1+1)))
print(mean(rbeta(N, aa2+1,ba2) > rbeta(N,aa2,ba2+1)))

print(mean(rbeta(N, ab1+1,bb1) > rbeta(N,ab1,bb1+1)))
print(mean(rbeta(N, ab2+1,bb2) > rbeta(N,ab2,bb2+1)))

