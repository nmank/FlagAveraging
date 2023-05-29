function [ MMG_array ] = make_data_MMG(d, l, n)
% Make a series of $n$ data elements from O(d)XO(l) \semi\prod M(d,l)
%
% NS, June 17

MMG_array = cell(n,3); 
for j=1:n
    MMG_array{j,1} = make_data_O_d(d,1);
    MMG_array{j,2} = make_data_O_d(l,1);
    MMG_array{j,3} = rand(d,l);
 end

end