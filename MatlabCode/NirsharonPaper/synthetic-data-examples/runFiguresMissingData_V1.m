% script name: "runFiguresMissingData_V1"

% scenarios: missing data, with fixed or vary noise (EIG-based)

% VERSION AUGUST 17, NS
% main change: adding LS and automatic choosing lambda

clear; clc;

d  = 5;
n1 = 100;
n2 = 300;

% FIXED NOISE, vary missing data
tic
n = n1;
[err1, snr1, dom, lambda_vals1] = MakeFigure_MissingDataWfixedNoise_V1(n1, d);
nameit = ['MissingData_n',num2str(n),'d',num2str(d),'V1'];
QuickSaveMissingDataFigure_V1(nameit, [min(dom), max(dom)]);
runtime1 = toc
nameit = ['justSPEC_CONT_',nameit];
Just_Spec_Cont_graph(dom, err1, n1, d, nameit ,1)


tic
n=n2;
[err2, snr2, dom, lambda_vals2] = MakeFigure_MissingDataWfixedNoise_V1(n2, d);
nameit = ['MissingData_n',num2str(n),'d',num2str(d),'V1'];
QuickSaveMissingDataFigure_V1(nameit, [min(dom), max(dom)]);
runtime2 = toc
save('data_Figure3_partial');
nameit = ['justSPEC_CONT_',nameit];
Just_Spec_Cont_graph(dom, err2, n2, d, nameit ,1)


% FIXED MISSING DATA, vary noise
tic
[err3, snr3, lambda_vals3] = MakeFigure_MissingDataWvaryNoise_V1(n1, d);
xrange = [(min(snr3)), max(snr3)];
nameit = ['MissingDataWnoise_n',num2str(n1),'d',num2str(d),'V1'];
QuickSaveMissingDataFigure_V1(nameit, xrange);
runtime3 = toc

nameit = ['justSPEC_CONT_',nameit];
Just_Spec_Cont_graph(snr3, err3, n1, d, nameit ,0)


tic
[err4, snr4, lambda_vals4] = MakeFigure_MissingDataWvaryNoise_V1(n2, d);
xrange = [(min(snr4)), max(snr4)];
nameit = ['MissingDataWnoise_n',num2str(n2),'d',num2str(d),'V1'];
QuickSaveMissingDataFigure_V1(nameit, xrange);
runtime4 = toc

nameit = ['justSPEC_CONT_',nameit];
Just_Spec_Cont_graph(snr4, err4, n2, d, nameit ,0)

save('data_Figure3');
