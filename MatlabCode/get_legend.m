function [hLeg] = get_legend(hFig)

h = get(hFig,'children');
hLeg = [];
for k = 1:length(h)
    if strcmpi(get(h(k),'Tag'),'legend')
        hLeg = h(k);
        break;
    end
end

end