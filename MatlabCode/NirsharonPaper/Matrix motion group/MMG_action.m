function [ prodMMG ] = MMG_action(g1, g2)
% Implementation of MMG action. g1 and g2 are elements, that is cell arrays
% of size 1X3

prodMMG = cell(1,3);
% A homoginuous product
prodMMG{1,1} = g1{1,1}*g2{1,1};
prodMMG{1,2} = g1{1,2}*g2{1,2};
% The semi-product part
prodMMG{1,3} = g1{1,3}+ g1{1,1}*g2{1,3}*g1{1,2}';

end

