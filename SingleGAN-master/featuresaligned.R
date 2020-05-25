cat(getwd())
cat('\n change to')
#setwd('/home/run/LYJ/R/LYJ')
#setwd('C:/Users/lyj/Dropbox/R/LYJ')
cat(getwd())
param<-commandArgs(T)
param
library(stringr)
library(readr)
data=XXX
data=as.data.frame(data)
row.names(data)=as.character(data$ID)
data=data[,-c(1,16:23)]
num=ncol(data)
stat=function(x1,y1){
  xx=mapply(function(x,y){
    res=wilcox.test(x,y,exact=F) 
    #res=ks.test(x,y,exact=F) 
    return(res$p.value) },x1,y1)
  sum(xx>0.05)
}
tuneGeTest=stat(  data[tuneGE,],data[tuneTarget,])
tunePhiTest=stat(  data[tunePhilps,],data[tuneTarget,])
tuneSieTest=stat(  data[tuneSiemens,],data[tuneTarget,])


modelName=as.vector(str_split( param,'/',simplify = T))
modelName=modelName[length(modelName)]
subjectsMain=sprintf('path:%s tuneGeTest:%.3f tunePhiTest:%.3f tuneSieTest:%.3f 
                 \n',
                 modelName,tuneGeTest,tunePhiTest,
                 tuneSieTest)
tuneGeTest=stat(  data[tuneGE,],oldData[tuneTarget,])
tunePhiTest=stat(  data[tunePhilps,],oldData[tuneTarget,])
tuneSieTest=stat(  data[tuneSiemens,],oldData[tuneTarget,])


subjects=sprintf('%scompare2Raw tuneGeTest:%.3f tunePhiTest:%.3f tuneSieTest:%.3f 
                 \n',
                 subjectsMain, tuneGeTest,tunePhiTest,
                 tuneSieTest)

flag=0;
while(flag<5)
{
  if(file.access('result_feature.txt',0)==0)
  {
    if(file.access('result_feature.txt',2)==0)
    {
      write.table(subjects,'result_feature.txt',
                  append = T,row.names = F,col.names = F);
      flag=10;
    }else
    {
      flag=flag+1;
      Sys.sleep(1);
    }
  }else
  {
    write.table(subjects,'result_feature.txt',
                append = T,row.names = F,col.names = F);
    flag=10;
  }
}
