% script name: "runFiguresOutliers"

% scenarios:outliers, with and without added noise, using LUD (both in separation and contraction)
clear; clc;

d=2;
n1 = 100;
n2 = 300;

n = n1;
MakeFigure_Outliers(n1, d)
nameit = ['Outliers_n',num2str(n),'d',num2str(d),'V1'];
QuickSaveOutliersFigure( nameit );

MakeFigure_OutliersWnoise(n1, d)
nameit = ['OutliersWnoise_n',num2str(n),'d',num2str(d),'V1'];
QuickSaveOutliersFigure( nameit );

n=n2;
MakeFigure_Outliers(n2, d)
nameit = ['Outliers_n',num2str(n),'d',num2str(d),'V1'];
QuickSaveOutliersFigure( nameit );

MakeFigure_OutliersWnoise(n2, d)
nameit = ['OutliersWnoise_n',num2str(n),'d',num2str(d),'V1'];
QuickSaveOutliersFigure( nameit );
