function [  ] = QuickSaveMissingDataFigure_V2(nameit, xrange, err1)

if nargin<3 
    err1 = [];
end

is_xrange = 1;
if nargin<2
    is_xrange = 0;
    xrange = [0.05,0.5];
end

set(gca,'FontSize',27);
if is_xrange
    set(gca,'xlim',xrange);
end
if size(err1,2)==5
    h_legend= legend('Least Squares','Separation','Contraction','Spectral','Contraction (MLE)','Location','northeast');
else
    h_legend= legend('Least Squares','Separation','Contraction','Spectral','Location','northeast');
end
set(h_legend,'FontSize',27);

if isempty(err1)
    ylim([-inf,1.5]);
else
    ylim([min(min(err1)),1.5]);
end

%set(gcf, 'Position', get(0,'ScreenSize'));
%set(gca,'xtick',floot(xrange):ceil(xrange))

saveas(gcf,nameit,'fig');        
saveas(gcf,nameit,'pdf');
print('-depsc2',nameit);
end

