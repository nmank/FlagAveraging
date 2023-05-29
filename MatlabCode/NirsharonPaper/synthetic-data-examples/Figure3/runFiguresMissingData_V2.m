% script name: "runFiguresMissingData_V2"
% 
% This script forms Figure 3 (with enough repeated iterations) 
% Scenarios: missing data, with fixed or vary noise 

% VERSION December 17, NS
% main changes: correcting the noise model (Guassian)
%               adding a spectral method with scaling

%( previous main change: adding LS and automatic choosing lambda)

clear; clc;

n1 = 100;
d1 = 3;

n2 = 200;
d2 = 5;

% FIXED NOISE, vary missing data
tic
n = n1;
[err1, snr1, dom1, lambda_vals1] = MakeFigure_MissingDataWfixedNoise_V2(n1, d1);

nameit = ['Missing_n',num2str(n1),'d',num2str(d1),'V2'];
QuickSaveMissingDataFigure_V2(nameit, [min(dom1), max(dom1)], err1);
runtime1 = toc


% FIXED MISSING DATA, vary noise
tic
[err3, snr3, lambda_vals3, AvailData3] = MakeFigure_MissingDataWvaryNoise_V2(n1, d1);

xrange = [(min(snr3)), max(snr3)];
nameit = ['Fixed_', num2str(100*AvailData3) ,'_percent_W_Vary_noise_n',num2str(n1),'d',num2str(d1),'V2'];
QuickSaveMissingDataFigure_V2(nameit, xrange, err3);
runtime3 = toc

%
tic
[err4, snr4, lambda_vals4] = MakeFigure_MissingDataWvaryNoise_V2(n2, d2);
xrange = [(min(snr4)), max(snr4)];
nameit = ['MissingW_varynoise_n',num2str(n2),'d',num2str(d2),'V2'];
QuickSaveMissingDataFigure_V2(nameit, xrange);
runtime4 = toc

save('data_Figure3_v2');
