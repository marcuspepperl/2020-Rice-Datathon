setwd('C:/Rice 2019-2020/Datathon')
library('readxl')
divorce = read_excel('C:/Rice 2019-2020/Datathon/divorce.xlsx')
divorce
divorced <- divorce[c(1:84), c(1:54)]
married <- divorce[c(85:170), c(1:54)]
divorced
married
divorced_means <- colMeans(divorced)
married_means <- colMeans(married)
names(divorced) <- c(1:54)
names(married) <- c(1:54)
par(mfrow=c(1,2))
plot(x = names(divorced), y = divorced_means, type = "p", xlim =c(0,55),
     ylim = c(0,4), col = "red",
     main = "Divorced Couples' Attribute Ratings
 (Divorced = Red, Married = Blue)", 
     xlab = "Attribute", 
     ylab = "Rating(0 being best)")
points(x = names(married), y = married_means, type = "p", col = "blue",
     main = "Married Couples' Attribute Ratings", xlab = "Attribute", 
     ylab = "Rating(0 being best)")
weights = read.table('Weights.txt')
Weights <- t(weights)
plot(x = c(1:54), y = Weights, type = "p", xlim = c(0, 55), 
     ylim = c(-.1, .15), main = "Attribute Influence",
     xlab = "Attribute", ylab = "Weight")
Influence <- (divorced_means - married_means) * Weights
plot(x = c(1:54), y = Influence, type = "p", xlim = c(0, 55),
     ylim = c(-.2, .5), main = "Overall Attribute Influence",
     xlab = "Attribute", ylab = "Divorce Index Contribution")
