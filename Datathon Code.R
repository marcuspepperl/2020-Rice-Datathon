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
divorced1 <- as.vector(t(divorced[,1]))
married1 <- as.vector(t(married[,1]))
t.test(divorced1, married1)
var(divorced1)
var(married1)
