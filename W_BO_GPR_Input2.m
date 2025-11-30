clc;
clear;
close all;
%% Preparing Input Data
% Uploading Data
ET= xlsread('Hailisu','M2:M3653')';
Rs= xlsread('Hailisu','N2:N3653')';
Tmean= xlsread('Hailisu','D2:D3653')'; 
RH= xlsread('Hailisu','F2:F3653')';

% Applying Delay to data
% Delays for time series forecating g(t)=f(g(t-1),g(t-2),...)
Delays=[1 2 3 4 5];    % 1-d ahead forecast
% Delays=[3 4 5 6 7];    % 3-d ahead forecast
% Delays=[5 6 7 8 9];    % 5-d ahead forecast
% Delays=[7 8 9 10 11];  % 7-d ahead forecast
%% wavelet ET
nLevel=3;   % Level of Decomposition
[a, d]=GetDWT(ET,nLevel,'db10');  % Extract wavelet according to mother wavelet (db10 in this code)

nx=numel(ET);
MaxDelay=max(Delays);
Range=(MaxDelay+1):nx;

ETInputs=[];
c=0;
for i=1:numel(Delays)  
%     Inputs(i,:)=x(Range-Delays(i));
    
    for k=1:nLevel
        c=c+1;
        ETInputs(c,:)=a{3}(Range-Delays(i));
        c=c+1;
        ETInputs(c,:)=d{k}(Range-Delays(i));
    end
    
   end
ETInputs=unique(ETInputs,'rows');

%% wavelet Rs
[a, d]=GetDWT(Rs,nLevel,'db10');
RsInputs=[];
c=0;
for i=1:numel(Delays)  
%     Inputs(i,:)=x(Range-Delays(i));
    
    for k=1:nLevel
        c=c+1;
        RsInputs(c,:)=a{3}(Range-Delays(i));
        c=c+1;
        RsInputs(c,:)=d{k}(Range-Delays(i));
    end
    
   end
RsInputs=unique(RsInputs,'rows');

%% wavelet Tmean
[a, d]=GetDWT(Tmean,nLevel,'db10');
TmeanInputs=[];
c=0;
for i=1:numel(Delays)  
%     Inputs(i,:)=x(Range-Delays(i));
    
    for k=1:nLevel
        c=c+1;
        TmeanInputs(c,:)=a{3}(Range-Delays(i));
        c=c+1;
        TmeanInputs(c,:)=d{k}(Range-Delays(i));
    end
    
   end
TmeanInputs=unique(TmeanInputs,'rows');

%% wavelet RH
[a, d]=GetDWT(RH,nLevel,'db10');
RHInputs=[];
c=0;
for i=1:numel(Delays)  
%     Inputs(i,:)=x(Range-Delays(i));
    
    for k=1:nLevel
        c=c+1;
        RHInputs(c,:)=a{3}(Range-Delays(i));
        c=c+1;
        RHInputs(c,:)=d{k}(Range-Delays(i));
    end
    
   end
RHInputs=unique(RHInputs,'rows');

%% Target
Targets = ET(Range);
Te=numel(Targets);
NY=3; % Test Period (Last 3 Years)
TestPeriod = 365*NY; 

%% Determining Train and test Data
Column = Te-TestPeriod;

trainETInputs=  ETInputs(:,(1:Column));
trainRsInputs=  RsInputs(:,(1:Column));
trainTmeanInputs=  TmeanInputs(:,(1:Column));
trainRHInputs=  RHInputs(:,(1:Column));

xTrainInputs= [trainETInputs;trainRsInputs;...
               trainTmeanInputs;trainRHInputs];

trainTargets= Targets(:,(1:Column));

testETInputs=  ETInputs(:,(Column+1:Te));
testRsInputs=  RsInputs(:,(Column+1:Te));
testTmeanInputs=  TmeanInputs(:,(Column+1:Te));
testRHInputs=  RHInputs(:,(Column+1:Te));

xTestInputs= [testETInputs;testRsInputs;...
              testTmeanInputs;testRHInputs];

testTargets= Targets(:,(Column+1:Te));

%% Determining Train and test Data
% For test
testInputsww=  xTestInputs;
testTargetsww= testTargets;

x = xTrainInputs';
ETobs= trainTargets';

%% Cross-Validation
NO= length(xTrainInputs)
c = cvpartition(NO,'Kfold',5)

%% Variables to Optimize
Var1 = optimizableVariable('Var1',[0.01,6],'Type','real','Transform','log');
Var2 = optimizableVariable('Var2',[0.01,6],'Type','real','Transform','log');
Var3 = optimizableVariable('Var3',[0.001,1],...
     'Type','real','Transform','log');
%% Objective Function

MinFn = @(z)kfoldLoss(fitrgp(x,ETobs,'CVPartition',c,'Basis','constant',...
    'FitMethod','SD','PredictMethod','exact',...
    'KernelFunction','squaredexponential','KernelParameters',...
    [z.Var1,z.Var2],'Sigma',z.Var3));

%% Perform Bayesian Optimization 
results = bayesopt(MinFn,[Var1,Var2,Var3],'MaxObj',20,...
    'AcquisitionFunctionName','expected-improvement-plus')

%% Evaluate Final Network
z(1) = results.XAtMinObjective.Var1;
z(2) = results.XAtMinObjective.Var2; 
z(3) = results.XAtMinObjective.Var3; 

gprMdl2 = fitrgp(x,ETobs,'Basis','constant',...
      'FitMethod','SD','PredictMethod','exact','KernelFunction',...
      'squaredexponential','KernelParameters',[z(1),z(2)],...
      'Sigma',z(3))
  
    trainOutputsgpr = predict(gprMdl2,x);
    testOutputsgpr = predict(gprMdl2,testInputsww');

save Results