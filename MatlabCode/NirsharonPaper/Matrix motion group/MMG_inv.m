function [ g_inv ] = MMG_inv(g)
% Implementation of MMG inverse, g is a cell array
% of size 1X3

g_inv = cell(1,3);
% A homoginuous inverse in O(d)
g_inv{1,1} = g{1,1}';
g_inv{1,2} = g{1,2}';
% The semi-product part
g_inv{1,3} = -g{1,1}'*g{1,3}*g{1,2};

end

