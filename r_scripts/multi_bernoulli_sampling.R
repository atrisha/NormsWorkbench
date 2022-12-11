library(mipfp)
library(questionr)
library(effectsize)

#or <- matrix(c(1,0.7,-0.5,0.65,
#               0.7,1,0.3,0.6,
#               -0.5,0.3,1,-0.4,
#               0.65,0.6,-0.4,1), nrow = 4, ncol = 4, byrow = TRUE)
or <- matrix(c(1     ,    -0.82362573,  0.18925517 , 0.22434411,
               -0.82362573 , 1  ,        0.08283328 ,-0.36079146,
               0.18925517,  0.08283328,  1   ,      -0.33339261,
               0.22434411 ,-0.36079146 ,-0.33339261 , 1        ), nrow = 4, ncol = 4, byrow = TRUE)

p <- c(0.41788752, 0.4858471,  0.2521992,  0.88520093)
f <- function(or,p){
  #or <- Corr2Odds(or,marg.probs=p)$odds
  rownames(or) <- colnames(or) <- c("n1", "n2", "n3", "n4")
  # hypothetical marginal probabilities
  
  
  # estimating the joint-distribution
  p.joint <- ObtainMultBinaryDist(corr = or, marg.probs = p)
  
  # simulating 100,000 draws from the obtained joint-distribution
  y.sim <- RMultBinary(n = 1e5, mult.bin.dist = p.joint)$binary.sequences
  
  # checking results
  cat('dim y.sim =', dim(y.sim)[1], 'x', dim(y.sim)[2], '\n')
  cat('Estimated marginal probs from simulated data\n')
  apply(y.sim,2,mean)
  cat('True probabilities\n')
  print(p)
  cat('Estimated correlation from simulated data\n')
  cor(y.sim)
  cat('True correlation\n')
  #Odds2Corr(or,p)$corr
  cat(str(p.joint$joint.proba))
  samples <- RMultBinary(n = 100, mult.bin.dist = p.joint)$binary.sequences
  samples <- data.frame(samples)
  return(samples)
}
#write.csv(samples,"C:\\Users\\Atrisha\\eclipse-workspace\\norms_workbench\\r_scripts\\samples.csv", row.names = FALSE, col.names = FALSE)


