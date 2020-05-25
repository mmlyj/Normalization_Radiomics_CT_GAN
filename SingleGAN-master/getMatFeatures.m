tic
addpath(genpath('/home/liyajun/data/v1.9'))
wavletUsing=[];
SelectedFeaturesMode=[1 1 1 1 1 1 1 ];
suffix=sprintf('%s_%s',modelName,epoch);
roiFormat='*.mat';
patientsList = subdir(fullfile(dataPath,roiFormat)); % 读取病例
newx=1;
newy=1;
newz=1;
is3D=false;
num=length(patientsList);

patientsList = subdir(fullfile(dataPath,roiFormat)); % ��ȡ����
num=length(patientsList);
[results,~]=prepareDeal(dataPath,is3D,wavletUsing,resultSavePath,SelectedFeaturesMode,false,1,[],'featureName_98.csv');
results = zeros(length(patientsList), size(results,2));
rangeData=true;
Resize=true;
minValue=-200;
maxValue=250;
for  i=1:length(patientsList)
    try
         id=patientsList(i).name;                                           %��һ��ΪID��
        disp(strcat(id,' dealing.'));
          ID=strsplit(id,'.');
        ID=strsplit(ID{1},filesep);
        ID=ID{end};
        if ~strcmp(suffix, '')
            ID=strsplit(ID,'_');
            ID=ID{1};
        end
        imgPath=id;
        res=load(imgPath);
        img=double( mat2gray(res.img));
        roiPath2=subdir(fullfile(roiPath,strcat(ID,roiFormat)));
        res=load(roiPath2.name);
        roi=res.roi;
        info=res.dicomInfo;
        pixelSapce=info.PixelSpacing;                
        roi=rmSmallConnponent(roi,10);
        sliceThickness=1;
        if Resize
            [img,roi, pixelSapce,sliceThickness] = ResizeSeries( img, roi, pixelSapce,sliceThickness,newx,newy,newz,0);
        end
        if rangeData            
            roi=outliersRange(img,roi,(minValue+200)/500,(maxValue+200)/500);%%ʹ��roi��ֵ���ų�����������
        end
        binmode=0;%%�̶�����
        fdisp=16;%%�Ҷȵȼ�16        
        wavemode=0;    
        feature=feature_caculate_2D(img,roi ,pixelSapce,wavletUsing,SelectedFeaturesMode,wavemode,binmode,fdisp);
      
        results(i,:)=[str2num(ID)  feature];
        disp(strcat(id,' done. ',num2str(i),' done.'));
    catch excption
        disp(excption);
        for j = 1:length(excption.stack)
            disp(excption.stack(j));
        end
        
        fprintf('error found in %s patients',id);
    end
end
if rangeData
    name=strcat('res_num',num2str(length(patientsList)),'_min',num2str(minValue),'_max',...
    num2str(maxValue),'_',suffix,'.csv');
else
      name = strcat('res_num',num2str(length(patientsList)),'_min',num2str(-200),'_max',...
       num2str(300),'_',suffix,'.csv');
end
if ~Resize
    name=strcat('noRize',name);
end
if  resultSavePath(length(resultSavePath))==filesep
    csvPath=[resultSavePath name];
else
    csvPath=[resultSavePath  filesep name];
end
if  exist(csvPath)==2
    delete(csvPath);
end
dlmwrite(csvPath,results,'-append','delimiter',',','precision',12);
check_aligned=1
  if check_aligned
   Command = {'Rscript','featuresaligned.R',csvPath};
else
  Command = {'Rscript','geResult.R',csvPath};
end
  system(strjoin(Command),'-echo');
  toc
  quit

