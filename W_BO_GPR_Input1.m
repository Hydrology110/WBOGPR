clc;
clear;
close all;
%% Preparing Input Data
tic % For recording the implementation time

% uploading Data from input file: Changde, Hailisu, Lancang, Miyun,
% Nanmulin, and Qiemo
x= xlsread('Hailisu','M2:M3653')'; 
                                                                
%% wavelet
nLevel=3;   % Level of Decomposition
[a, d]=GetDWT(x,nLevel,'db10');  % Extract wavelet according to mother wavelet (db10 in this code)

nx=numel(x);

% Number of lag days should be calculated based on PACF results (Fig. 3)
% For Hailisu the optimum lag number is 13. 
Delays= [1 2 3 4 5 6 7 8 9 10 11 12 13]; % 1-d ahead forecast  
% Delays= [3 4 5 6 7 8 9 10 11 12 13];     % 3-d ahead forecast
% Delays= [5 6 7 8 9 10 11 12 13];         % 5-d ahead forecast
% Delays= [7 8 9 10 11 12 13];             % 7-d ahead forecast

MaxDelay=max(Delays);

Range=(MaxDelay+1):nx;

Inputs=[];
c=0;
for i=1:numel(Delays)  
%     Inputs(i,:)=x(Range-Delays(i));
    
    for k=1:nLevel
        c=c+1;
        Inputs(c,:)=a{3}(Range-Delays(i));
        c=c+1;
        Inputs(c,:)=d{k}(Range-Delays(i));
    end
    
   end
Inputs=unique(Inputs,'rows');

Targets=x(Range);
Te=numel(Targets);
NY=3; % Test Period (Last 3 Years)
TestPeriod = 365*NY; 
Tepercent=(TestPeriod/Te);  
Trpercent=(1-(Tepercent));

%% Determining Train and test Data
Column = Te-TestPeriod;      

trainInputsww=  Inputs(:,(1:Column));
trainTargetsww= Targets(:,(1:Column));

% For test
TT= Te-TestPeriod;
testInputsww=  Inputs(:,(Column+1:Te));
testTargetsww= Targets(:,(Column+1:Te));

x = trainInputsww';
ETobs= trainTargetsww';

%% Cross-Validation
NO= length(trainInputsww)
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
   
PlotResults(testTargetsww,testOutputsgpr','Test Data GPR');
toc
save results