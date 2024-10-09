clc;close all;clear;%delete(findall(0));
%% 
N=40;
for s=1:N
    a = find(docfile.trai == s);
    out(s,1) = s; 
    out(s,2) = mean(docfile.VarName8(min(a):max(a)));
    out(s,3) = mean(docfile.VarName10(min(a):max(a)));
    out(s,4) = mean(docfile.VarName12(min(a):max(a)));
    out(s,5) = mean(docfile.VarName14(min(a):max(a)));
    
end

%% Splitting dataset val
is_to_be_deleted = false( size(Signals_Struct.signal,1), 1 );
n=0;
for kk = 1:10:size(Signals_Struct.signal,1)
    n=n+1;
    Signals_Struct_val.signal(n,:) = Signals_Struct.signal(kk,:);
    Signals_Struct_val.Labels(n,:) = Signals_Struct.Labels(kk,:);
    Signals_Struct_val.CI(n,:) = Signals_Struct.CI(kk,:);
    is_to_be_deleted(kk) = true;
end
%% splitting dataset test
n=0; 
for kk = 3:5:size(Signals_Struct.signal,1)
    n=n+1;
    Signals_Struct_test.signal(n,:) = Signals_Struct.signal(kk,:);
    Signals_Struct_test.Labels(n,:) = Signals_Struct.Labels(kk,:);
    Signals_Struct_test.CI(n,:) = Signals_Struct.CI(kk,:);
    is_to_be_deleted(kk) = true;
end
Signals_Struct.signal( is_to_be_deleted, : ) = [];
Signals_Struct.Labels( is_to_be_deleted, : ) = [];
Signals_Struct.CI( is_to_be_deleted, : ) = [];
%% image genrate for test and val
sig =struct.ecg(1,:); 
Number_of_current_image=0;
signalLength = length(sig);
Fs = 40.9;
fb = cwtfilterbank('SignalLength',signalLength,'SamplingFrequency',Fs)

fb = cwtfilterbank('SignalLength',signalLength,'Wavelet','amor','SamplingFrequency',Fs,'VoicesPerOctave',48)
[wt,f] = fb.wt(sig);
[r,c] = size(struct.ecg);
lab = struct.info(:,1);
%% lebels correction
clearvars labels
for ii=1:length(lab) 
    if lab(ii,1)== 3
       labels(Number_of_current_image+ii,1) = {'Hypopnea'}; 
    end
    if lab(ii,1)== 2
       labels(Number_of_current_image+ii,1) = {'Apnea'};
    end
end
%% file directory

ret = exist('E:\shhs2\normal vs apnea vs hypopnea','dir');
if ret ==0
    mkdir('E:\shhs2\normal vs apnea vs hypopnea');
end
tic
k=0;
for ii=Number_of_current_image+1:Number_of_current_image+r
    k=k+1;
    [wt,~] = fb.wt(struct.ecg(k,:));
    imageRoot = fullfile(pwd,'image');
    im = ind2rgb(im2uint8(rescale(abs(wt))),jet(256));
    imgLoc = fullfile(imageRoot,char(labels(ii)));
    imFileName = strcat(num2str(ii),'.jpg');
    imwrite(imresize(im,[512 512]),fullfile(imgLoc,imFileName));   
end
toc
%% EMD separation
tic
nn = 0;
for ii = 1:2000%size(Signals_Struct.signal,1) 
    EMD = emd(Signals_Struct.signal(ii,:));
    for bb= 1:size(EMD,2)
        r = corrcoef(EMD(:,bb),Signals_Struct.signal(ii,:));
        if r(2) > .6
            nn=nn+1;
            Signals_Struct_EMD.signal(nn,:)=EMD(:,bb);
            Signals_Struct_EMD.Labels(nn,:)=Signals_Struct.Labels(ii,:);
        end
    end
end
toc
for ii = 2001:4000%size(Signals_Struct.signal,1) 
    EMD = emd(Signals_Struct.signal(ii,:));
    for bb= 1:size(EMD,2)
        r = corrcoef(EMD(:,bb),Signals_Struct.signal(ii,:));
        if r(2) > .6
            nn=nn+1;
            Signals_Struct_EMD.signal(nn,:)=EMD(:,bb);
            Signals_Struct_EMD.Labels(nn,:)=Signals_Struct.Labels(ii,:);
        end
    end
end

for ii = 4001:6000%size(Signals_Struct.signal,1) 
    EMD = emd(Signals_Struct.signal(ii,:));
    for bb= 1:size(EMD,2)
        r = corrcoef(EMD(:,bb),Signals_Struct.signal(ii,:));
        if r(2) > .6
            nn=nn+1;
            Signals_Struct_EMD.signal(nn,:)=EMD(:,bb);
            Signals_Struct_EMD.Labels(nn,:)=Signals_Struct.Labels(ii,:);
        end
    end
end

for ii = 6001:9000%size(Signals_Struct.signal,1) 
    EMD = emd(Signals_Struct.signal(ii,:));
    for bb= 1:size(EMD,2)
        r = corrcoef(EMD(:,bb),Signals_Struct.signal(ii,:));
        if r(2) > .6
            nn=nn+1;
            Signals_Struct_EMD.signal(nn,:)=EMD(:,bb);
            Signals_Struct_EMD.Labels(nn,:)=Signals_Struct.Labels(ii,:);
        end
    end
end

for ii = 9001:11368 %size(Signals_Struct.signal,1) 
    EMD = emd(Signals_Struct.signal(ii,:));
    for bb= 1:size(EMD,2)
        r = corrcoef(EMD(:,bb),Signals_Struct.signal(ii,:));
        if r(2) > .6
            nn=nn+1;
            Signals_Struct_EMD.signal(nn,:)=EMD(:,bb);
            Signals_Struct_EMD.Labels(nn,:)=Signals_Struct.Labels(ii,:);
        end
    end
end
toc
%% EMD image genarate
sig =Signals_Struct_EMD.signal(1,:);
signalLength = length(sig);
Fs = 100;
fb = cwtfilterbank('SignalLength',signalLength,'SamplingFrequency',Fs)

fb = cwtfilterbank('SignalLength',signalLength,'Wavelet','amor','SamplingFrequency',Fs,'VoicesPerOctave',48)
[wt,f] = fb.wt(sig);
[r,c] = size(Signals_Struct_EMD.signal);
labels = Signals_Struct_EMD.Labels;
ret = exist('E:\using alex net\2653129-Code-TransferLearningExample\8.ECG apnea data','dir');
if ret ==0
    mkdir('E:\using alex net\2653129-Code-TransferLearningExample\8.ECG apnea data');
end
for ii = 1:r
    [wt,~] = fb.wt(Signals_Struct_EMD.signal(ii,:));
    imageRoot = fullfile(pwd,'EMD .7');
    im = ind2rgb(im2uint8(rescale(abs(wt))),jet(128));
    imgLoc = fullfile(imageRoot,char(labels(ii)));
    imFileName = strcat(num2str(ii),'.jpg');
    imwrite(imresize(im,[32 32]),fullfile(imgLoc,imFileName));   
end
clear Signals_Struct_EMD
%% CNN
%load images
% for train data
digitDatasetPath1 = fullfile('E:\using alex net\2653129-Code-TransferLearningExample\Apnea journal data\Hybrid\train - Copy (2)');
trainimg = imageDatastore(digitDatasetPath1, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
% for valivadation data
digitDatasetPath2 = fullfile('E:\using alex net\2653129-Code-TransferLearningExample\Apnea journal data\normal\val');
valimg = imageDatastore(digitDatasetPath2, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
% for test data
digitDatasetPath3 = fullfile('E:\using alex net\2653129-Code-TransferLearningExample\Apnea journal data\normal\test');
testimg = imageDatastore(digitDatasetPath3, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
% Determine the split
%[trainimg valimg testimg] = splitEachLabel(imds,.70,.1,.2,'randomize');
%total_split=countEachLabel(imds);
% Number of Images
train_images=length(trainimg.Labels);
val_images=length(valimg.Labels);
test_images=length(testimg.Labels);

%%K-fold Validation
% Number of folds
%for mm=1:1
lr = 0.0004;

plt = 'training-progress'
mx=130;
mb=16;
options = trainingOptions('sgdm',...
    'ExecutionEnvironment','gpu',...
    'MaxEpochs',mx,'MiniBatchSize',mb,...
    'Shuffle','every-epoch', ...
    'InitialLearnRate',lr, ...
    'Verbose',true, ...
    'Plots',plt, ...
    'Plots','training-progress',...
    'ValidationData',valimg);

trained_net = trainNetwork(trainimg,Model40,options);

sprintf('------------Validation result----------')
predictedvalLabels = classify(trained_net,valimg);
clear real_val predict_val
for i=1:length(valimg.Labels)
    if isequal(valimg.Labels(i),{'Apnea'})
        real_val(i)= 1;
    end
    if isequal(predictedvalLabels(i),{'Apnea'})
        predict_val(i)= 1;
    end
    if isequal(valimg.Labels(i),{'Normal'})
        real_val(i)= 0;
    end
    if isequal(predictedvalLabels(i),{'Normal'})
        predict_val(i)= 0;
    end
end
[c_matrix,Result,RefereceResult]= confusion.getMatrix(real_val,predict_val);
clear Totalresult AHI_pred AHI_real
Totalresult(1) = Result.Accuracy;
Totalresult(3) = Result.Sensitivity;
Totalresult(5) = Result.Specificity;
Totalresult(7) = Result.F1_score;
Totalresult(9) = Result.Precision;
sprintf('------------Test result----------')
clear real_val predict_val
predictedvalLabels = classify(trained_net,testimg);
for i=1:length(testimg.Labels)
    if isequal(testimg.Labels(i),{'Apnea'})
        real_val(i)= 1;
    end
    if isequal(predictedvalLabels(i),{'Apnea'})
        predict_val(i)= 1;
    end
    if isequal(testimg.Labels(i),{'Normal'})
        real_val(i)= 0;
    end
    if isequal(predictedvalLabels(i),{'Normal'})
        predict_val(i)= 0;
    end
end
[c_matrix,Result,RefereceResult]= confusion.getMatrix(real_val,predict_val);
Totalresult(2) = Result.Accuracy;
Totalresult(4) = Result.Sensitivity;
Totalresult(6) = Result.Specificity;
Totalresult(8) = Result.F1_score;
Totalresult(10) = Result.Precision;

%% per recording
cd('E:\using alex net\2653129-Code-TransferLearningExample\8.ECG apnea data\subject');
rrr=0; 
for mm=1:9
    digitDatasetPath = fullfile(sprintf('x0%d',mm));
    mash = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    testme = augmentedImageDatastore([32 32],mash);
    fazla = classify(trained_net, testme);
    outcome = countcats(fazla);
    AHI_pred(mm) = (outcome(1)/(outcome(2)+outcome(1)))*60;
    AHI_real(mm) =  CI.h(CI.ID == (sprintf('x0%d',mm)));
    if AHI_pred(mm) <5
        Predicted(mm) = 0;
    end
    if AHI_pred(mm) >5 | AHI_pred(mm) ==5
        Predicted(mm) = 1;
    end
    if CI.h(CI.ID == (sprintf('x0%d',mm))) <5
        Real(mm) = 0;
    end
    if CI.h(CI.ID == (sprintf('x0%d',mm))) >5 | CI.h(CI.ID == (sprintf('x0%d',mm))) == 5
        Real(mm) = 1;
    end
end
for mm=10:35
    digitDatasetPath = fullfile(sprintf('x%d',mm));
    mash = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    testme = augmentedImageDatastore([32 32],mash);
    fazla = classify(trained_net, testme);
    outcome = countcats(fazla);
    AHI_pred(mm) = (outcome(1)/(outcome(2)+outcome(1)))*60;
    AHI_real(mm) =  CI.h(CI.ID == (sprintf('x%d',mm)));
    if AHI_pred(mm) <5
        Predicted(mm) = 0;
    end
    if AHI_pred(mm) >5 | AHI_pred(mm) ==5
        Predicted(mm) = 1;
    end
    if CI.h(CI.ID == (sprintf('x%d',mm))) <5
        Real(mm) = 0;
    end
    if CI.h(CI.ID == (sprintf('x%d',mm))) >5 | CI.h(CI.ID == (sprintf('x%d',mm))) == 5
        Real(mm) = 1;
    end
end
nn=35;
for mm=1:9
    nn = nn+1;
    digitDatasetPath = fullfile(sprintf('a0%d',mm));
    mash = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    testme = augmentedImageDatastore([32 32],mash);
    fazla = classify(trained_net, testme);
    outcome = countcats(fazla);
    AHI_pred(nn) = (outcome(1)/(outcome(2)+outcome(1)))*60;
    AHI_real(nn) =  CI.h(CI.ID == (sprintf('a0%d',mm)));
    if AHI_pred(nn) <5
        Predicted(nn) = 0;
    end
    if AHI_pred(nn) >5 | AHI_pred(nn) ==5
        Predicted(nn) = 1;
    end
    if CI.h(CI.ID == (sprintf('a0%d',mm))) <5
        Real(nn) = 0;
    end
    if CI.h(CI.ID == (sprintf('a0%d',mm))) >5 | CI.h(CI.ID == (sprintf('a0%d',mm))) == 5
        Real(nn) = 1;
    end
end

for mm=10:20
    nn = nn+1;
    digitDatasetPath = fullfile(sprintf('a%d',mm));
    mash = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    testme = augmentedImageDatastore([32 32],mash);
    fazla = classify(trained_net, testme);
    outcome = countcats(fazla);
    AHI_pred(nn) = (outcome(1)/(outcome(2)+outcome(1)))*60;
    AHI_real(nn) =  CI.h(CI.ID == (sprintf('a%d',mm)));
    if AHI_pred(nn) <5
        Predicted(nn) = 0;
    end
    if AHI_pred(nn) >5 | AHI_pred(nn) ==5
        Predicted(nn) = 1;
    end
    if CI.h(CI.ID == (sprintf('a%d',mm))) <5
        Real(nn) = 0;
    end
    if CI.h(CI.ID == (sprintf('a%d',mm))) >5 | CI.h(CI.ID == (sprintf('a%d',mm))) == 5
        Real(nn) = 1;
    end
end

for mm=1:5
    nn = nn+1;
    digitDatasetPath = fullfile(sprintf('b0%d',mm));
    mash = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    testme = augmentedImageDatastore([32 32],mash);
    fazla = classify(trained_net, testme);
    outcome = countcats(fazla);
    AHI_pred(nn) = (outcome(1)/(outcome(2)+outcome(1)))*60;
    AHI_real(nn) =  CI.h(CI.ID == (sprintf('b0%d',mm)));
    if AHI_pred(nn) <5
        Predicted(nn) = 0;
    end
    if AHI_pred(nn) >5 | AHI_pred(nn) ==5
        Predicted(nn) = 1;
    end
    if CI.h(CI.ID == (sprintf('b0%d',mm))) <5
        Real(nn) = 0;
    end
    if CI.h(CI.ID == (sprintf('b0%d',mm))) >5 | CI.h(CI.ID == (sprintf('b0%d',mm))) == 5
        Real(nn) = 1;
    end
end

for mm=1:9
    nn = nn+1;
    digitDatasetPath = fullfile(sprintf('c0%d',mm));
    mash = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    testme = augmentedImageDatastore([32 32],mash);
    fazla = classify(trained_net, testme);
    outcome = countcats(fazla);
    AHI_pred(nn) = (outcome(1)/(outcome(2)+outcome(1)))*60;
    AHI_real(nn) =  CI.h(CI.ID == (sprintf('c0%d',mm)));
    if AHI_pred(nn) <5
        Predicted(nn) = 0;
    end
    if AHI_pred(nn) >5 | AHI_pred(nn) ==5
        Predicted(nn) = 1;
    end
    if CI.h(CI.ID == (sprintf('c0%d',mm))) <5
        Real(nn) = 0;
    end
    if CI.h(CI.ID == (sprintf('c0%d',mm))) >5 | CI.h(CI.ID == (sprintf('c0%d',mm))) == 5
        Real(nn) = 1;
    end
end
tic
for mm=10
    nn = nn+1;
    digitDatasetPath = fullfile(sprintf('c%d',mm));
    mash = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    testme = augmentedImageDatastore([32 32],mash);
    tic
    fazla = classify(trained_net, testme);
    toc
    outcome = countcats(fazla);
    AHI_pred(nn) = (outcome(1)/(outcome(2)+outcome(1)))*60;
    AHI_real(nn) =  CI.h(CI.ID == (sprintf('c%d',mm)));
    if AHI_pred(nn) <5
        Predicted(nn) = 0;
    end
    if AHI_pred(nn) >5 | AHI_pred(nn) ==5
        Predicted(nn) = 1;
    end
    if CI.h(CI.ID == (sprintf('c%d',mm))) <5
        Real(nn) = 0;
    end
    if CI.h(CI.ID == (sprintf('c%d',mm))) >5 | CI.h(CI.ID == (sprintf('c%d',mm))) == 5
        Real(nn) = 1;
    end
end
toc
[c_matrix,Result,RefereceResult]= confusion.getMatrix(Real,Predicted);

R = corrcoef(AHI_pred,AHI_real);
Totalresult(11) = Result.Accuracy;
Totalresult(12) = Result.Sensitivity;
Totalresult(13) = Result.Specificity;
Totalresult(14) = Result.F1_score;
Totalresult(15) = Result.Precision;
Totalresult(16) = R(2);
v = Real == Predicted;
n=17;
ll=17;
Totalresult(ll) = 999999;
for oo=1:70
    if v(oo) == 0
        n = n+1;
        Totalresult(n) = oo;
        n = n+1;
        Totalresult(n) = AHI_real(oo);
        n = n+1;
        Totalresult(n) = AHI_pred(oo);
        n = n+1;
        Totalresult(n) = 9999999;
    end
end
for uu=1:16
    Totalresult(uu) = Totalresult(uu)*100;
end

cd('E:\using alex net\2653129-Code-TransferLearningExample\Apnea journal data\Hybrid\trail\18')

save(sprintf('LR=%d_epoch=%d_batch=%d_drop=.1.mat',lr,mx,mb),'Model40','trained_net', 'AHI_pred', 'AHI_real','valimg','testimg','Totalresult');



%% withheld per-recoding
cd('E:\using alex net\2653129-Code-TransferLearningExample\8.ECG apnea data\subject');
rrr=0; 
for mm=[1]
    digitDatasetPath = fullfile(sprintf('x0%d',mm));
    mash = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    testme = augmentedImageDatastore([32 32],mash);
    fazla = classify(trained_net, testme);
    for i=1:length(fazla)
        if isequal(fazla(i),{'Apnea'})
            rrr=rrr+1;
            Persegment_result(rrr,1)= 1;    
        end
        if isequal(fazla(i),{'Normal'})
            rrr=rrr+1;
            Persegment_result(rrr,1)=0;   
        end
    end
end
for mm=17
    digitDatasetPath = fullfile(sprintf('x%d',mm));
    mash = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    testme = augmentedImageDatastore([32 32],mash);
    fazla = classify(trained_net, testme);
 for i=1:length(fazla)
        if isequal(fazla(i),{'Apnea'})
            rrr=rrr+1;
            Persegment_result(rrr,1)= 1;    
        end
        if isequal(fazla(i),{'Normal'})
            rrr=rrr+1;
            Persegment_result(rrr,1)=0;   
        end
    end
end
ooo=0;
for i=1:size(Signals_Struct.Labels,1)
        if isequal(Signals_Struct.Labels(i),{'Apnea'})
            ooo=ooo+1;
            Persegment_truth(ooo,1)= 1;    
        end
        if isequal(Signals_Struct.Labels(i),{'Normal'})
            ooo=ooo+1;
            Persegment_truth(ooo,1)=0;   
        end
    end
[c_matrix,Result,RefereceResult]= confusion.getMatrix(Persegment_truth,Persegment_result);

ll=15
    cd('E:\using alex net\2653129-Code-TransferLearningExample\8.ECG apnea data\ECG apnea data');
fid = fopen(sprintf('x%d.txt',ll), 'w');
if fid < 0; error('Could not open file because "%s"', msg); end
fprintf(fid, '%c \n', fileread(sprintf('x%dxx.txt',ll)));      
fclose(fid);

clear Persegment_truth Persegment_result Signals_Struct
[hdr, record]=edfread(sprintf('x%d.edf',ll));
% plot(record(1,:))
rawsignal = record(1,:);
N = floor((length(rawsignal)/100)/60);
n=0;i=0;jj=00; %size(Signals_Struct.signal,1);
for r=1:N
    for c=1:6000
        i=i+1;
        Signals_Struct.signal((r+jj),c) = rawsignal(1,i);
    end
      if isequal(x15.event(r,:),{'N'})
            Signals_Struct.Labels(r+jj,1) = 0; 
      end
      if isequal(x15.event(r,:),{'A'})
            Signals_Struct.Labels(r+jj,1) = 1;
      end
end 

%% 
for i=1:size(Signals_Struct.signal,1)
    Signals_Struct.signal(i,:) = filter(d1,Signals_Struct.signal(i,:));
end

is_to_be_deleted = false( size(Signals_Struct.signal,1), 1 );
for i=1:size(Signals_Struct.signal,1)
    a(1,:)= Signals_Struct.signal(i,1:1000);
    a(2,:)= Signals_Struct.signal(i,1001:2000);
    a(3,:)= Signals_Struct.signal(i,2001:3000);
    a(4,:)= Signals_Struct.signal(i,3001:4000);
    a(5,:)= Signals_Struct.signal(i,4001:5000);
    a(6,:)= Signals_Struct.signal(i,5001:6000);
    maximum = max(a');
    minimum = min(a');
    M = sum(maximum > 2*median(maximum));
    N = sum(minimum < 2*median(minimum));
    if (M > 0 | N >0 )
        is_to_be_deleted(i) = true;
    end
end
Signals_Struct.signal( is_to_be_deleted, : ) = [];
Signals_Struct.Labels( is_to_be_deleted, : ) = [];
% Signals_Struct.CI( is_to_be_deleted, : ) = [];



for i=1:size(Signals_Struct.signal,1)
    Signals_Struct.signal(i,:) = detrend(Signals_Struct.signal(i,:));
end

is_to_be_deleted = false( size(Signals_Struct.signal,1), 1 );
for i=1:size(Signals_Struct.signal,1)
    a(1,:)= Signals_Struct.signal(i,1:1000);
    a(2,:)= Signals_Struct.signal(i,1001:2000);
    a(3,:)= Signals_Struct.signal(i,2001:3000);
    a(4,:)= Signals_Struct.signal(i,3001:4000);
    a(5,:)= Signals_Struct.signal(i,4001:5000);
    a(6,:)= Signals_Struct.signal(i,5001:6000);
    V = sum(isnan(a'));
    maximum = max(a');
    minimum = abs(min(a'));
    A = abs(maximum + minimum);
    B = sum(A < .1 );
    if (V > 0 | B > 0)
        is_to_be_deleted(i) = true;
    end
end
Signals_Struct.signal( is_to_be_deleted, : ) = [];
Signals_Struct.Labels( is_to_be_deleted, : ) = [];
% Signals_Struct.CI( is_to_be_deleted, : ) = [];

rrr=0;
cd('E:\using alex net\2653129-Code-TransferLearningExample\8.ECG apnea data\subject');
 digitDatasetPath = fullfile(sprintf('x%d',ll));
    mash = imageDatastore(digitDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
    testme = augmentedImageDatastore([32 32],mash);
    fazla = classify(trained_net, testme);
    for i=1:length(fazla)
        if isequal(fazla(i),{'Apnea'})
            rrr=rrr+1;
            Persegment_result(rrr,1)= 1;    
        end
        if isequal(fazla(i),{'Normal'})
            rrr=rrr+1;
            Persegment_result(rrr,1)=0;   
        end
    end
    
    Persegment_truth= Signals_Struct.Labels;
    [c_matrix,Result,RefereceResult]= confusion.getMatrix(Persegment_truth,Persegment_result);
%% image genrate for test and val
sig =mash; 
Number_of_current_image=0;
signalLength = length(sig);
[r,c] = size(struct.ecg);
lab = struct.info(:,1);
%% lebels correction
clearvars labels
for ii=1:length(lab) 
    if lab(ii,1)== 3
       labels(Number_of_current_image+ii,1) = {'Hypopnea'}; 
    end
    if lab(ii,1)== 2
       labels(Number_of_current_image+ii,1) = {'Apnea'};
    end
end
%% file directory

ret = exist('E:\shhs2\normal vs apnea vs hypopnea','dir');
if ret ==0
    mkdir('E:\shhs2\normal vs apnea vs hypopnea');
end
tic
k=0;
for ii=1:409
    k=k+1;
fb = cwtfilterbank('SignalLength',signalLength,'SamplingFrequency',ii);

fb = cwtfilterbank('SignalLength',signalLength,'Wavelet','bump','SamplingFrequency',Fs,'VoicesPerOctave',48);
[wt,f] = fb.wt(sig);
    [wt,~] = fb.wt(mash);
    imageRoot = fullfile(pwd,'vua');
    im = ind2rgb(im2uint8(rescale(abs(wt))),jet(256));
    imgLoc = fullfile(imageRoot,char(trainLabels(k)));
    imFileName = strcat(num2str(ii),'.jpg');
    imwrite(imresize(im,[512 512]),fullfile(imgLoc,imFileName));   
end