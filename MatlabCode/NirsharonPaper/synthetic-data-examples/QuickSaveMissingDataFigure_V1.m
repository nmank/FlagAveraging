function [  ] = QuickSaveMissingDataFigure_V1(nameit, xrange)

is_xrange = 1;
if nargin<2
    is_xrange = 0;
    xrange = [0.05,0.5];
end

set(gca,'FontSize',27);
if is_xrange
    set(gca,'xlim',xrange);
end
h_legend= legend('Least Squares','Separation','Contraction','Spectral','Location','northeast');
set(h_legend,'FontSize',26);

saveas(gcf,nameit,'fig');        
saveas(gcf,nameit,'pdf');
print('-depsc2',nameit);
end

