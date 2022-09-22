# The following code to create a dataframe and remove duplicated rows is always executed and acts as a preamble for your script: 

# dataset <- data.frame(R_Value, R_score, F_Value, F_score, M_Value, M_score)
# dataset <- unique(dataset)

# Paste or type your script code here:
library(scatterplot3d)

x <- dataset$R_Value

y <- dataset$F_Value

z <- dataset$M_Value

scatterplot3d(x,y,z,
color=as.numeric(dataset$RFM),
pch=19,
xlab="R value",
ylab="F value",
zlab="M value",
angle=dataset$Angle[1]
)
# [model] > [new parameter] > set para and add para to 'values'

library(ggplot2)

data(“economics”)

x <- economics$date

y1 <- economics$psavert

y2 <- economics$unemploy

plot(x,y1,col=”red”,type=”l”,ylab=”Savings Rate %”, xlab=”Year”)

par(new=TRUE)

plot(x,y2,yaxt=”n”,xaxt=”n”,ylab=””,xlab=””,col=”blue”,type=”l”)

axis(side=4)

legend(“topleft”, c(“Savings Rate”,”Unemployed”),

   col=c(“red”,”blue”),lty=c(1,1))
