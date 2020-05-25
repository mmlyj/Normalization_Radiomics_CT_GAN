cat(getwd())
cat('\n change to')
#setwd('/home/run/LYJ/R/LYJ')
#setwd('C:/Users/lyj/Dropbox/R/LYJ')
cat(getwd())
param<-commandArgs(T)
param
library(readr)
library(readxl)
library(stringr)
#library(stringi)
###loadData###
allData=XXXX

row.names( allData)=as.character(allData$ID)
allData=subset(allData,select = -ID)
# GE=cbind(GEData[c(GEHighRisk,GElowRisk),c('status','time')],
#          allData[c(GEHighRisk,GElowRisk),] )
comm=intersect( row.names( allData),row.names(GEData))
GE=cbind(GEData[comm,c('status','time')],
         allData[comm,] )
print('GE')
nrow(GE)
comm1=intersect( row.names( allData),row.names(PhilipsData))
Philips=cbind(PhilipsData[comm1,c('status','time')],
              allData[comm1,] )
print('Philips')
nrow(Philips)
comm2=intersect( row.names( allData),row.names(SiemensData))
Siemens=cbind(SiemensData[comm2,c('status','time')],
              allData[comm2,] )
print('Siemens')
nrow(Siemens)
###loadData###



MRMRWrapped<-function(dealDataFrame,num)
{
  library(mRMRe)
  featuresNum<-num
  trainData<-as.data.frame(sapply(dealDataFrame,as.numeric))
  dd <- mRMR.data(data = trainData) # data input to mRMRe
  ft <- mRMR.classic(data = dd, target_indices = c(1),
                     feature_count = featuresNum)
  var<-colnames(trainData)[as.vector(ft@filters$`1`)]
  return(var)
}
testOnDiffer=function(trainData,testData){
  lambdaUsing=seq(0.01,0.2,0.002)
  trainData=trainData[,-c(17:24)]#[,c(1:2,17:24)] #rbind(Philips,GE)#[,c('y',wilcoxDiff)]
  testData=testData[,-c(17:24)]#[,c(1:2,17:24)]#[,c('y',wilcoxDiff)]
  #type.measure="auc"
  trainData=as.data.frame( cbind(y=trainData$status,trainData[,-c(1,2)]))
  testData=as.data.frame( cbind(y=testData$status,testData[,-c(1,2)]))
  varUsing=MRMRWrapped(trainData,20)
  trainData=trainData[,c('y',varUsing)]
  testData=testData[,c('y',varUsing)]
  y=trainData$y
  x=as.matrix(trainData[,-1])
  library(glmnet)
  res=rep(0,50)
  for (i in 1:50)
  {
    set.seed(i)#trainData
    cv=cv.glmnet(x,y,family="binomial",alpha=0,
                 nlambda = 300,nfolds=3,type.measure = 'auc')#,pmax=5)

    fphe <- predict(cv$glmnet.fit,newx=as.matrix( testData[,-1])
                    ,s=cv$lambda.min,type ='response')
    res[i]=pROC::auc(as.vector( testData$y),as.vector( fphe) )
    
    # fphe<- predict(cv$glmnet.fit,newx=as.matrix( trainData[,-1])
    #                ,s=cv$lambda.min,type ='response')
    # res[i]=pROC::auc(as.vector( trainData$y),as.vector( fphe) )
    # 
   
  }
  res=res[res!=0.5]
  mean(res)
}
GE2Siemens=testOnDiffer(GE,Siemens)
GE2Philips=testOnDiffer(GE,Philips)
Siemens2GE=testOnDiffer(Siemens,GE)
Siemens2Philips=testOnDiffer(Siemens,Philips)
Philips2Siemens=testOnDiffer(Philips,Siemens)
Philips2GE=testOnDiffer(Philips,GE)
differentStr=sprintf('GE2Siemens:%.3f GE2Philips:%.3f Siemens2GE:%.3f Siemens2Philips:%.3f Philips2Siemens:%.3f Philips2GE:%.3f',
                     GE2Siemens,GE2Philips,Siemens2GE,Siemens2Philips,
                     Philips2Siemens,Philips2GE)

modelName=as.vector(str_split( param,'/',simplify = T))
modelName=modelName[length(modelName)]
subjects=sprintf('path:%s Siemens:%.3f GE:%.3f Philips:%.3f combined:%.3f\n %s\n',
                 modelName,differentStr)

flag=0;
while(flag<5)
{
  if(file.access('result.txt',0)==0)
  {
    if(file.access('result.txt',2)==0)
    {
      write.table(subjects,'result.txt',
                  append = T,row.names = F,col.names = F);
      flag=10;
    }else
    {
      flag=flag+1;
      Sys.sleep(1);
    }
  }else
  {
    write.table(subjects,'result.txt',
                append = T,row.names = F,col.names = F);
    flag=10;
  }
}
