function [dist] = chordal_distance(data, Y, flag_type)

 if ~exist('flag_type','var')
      flag_type = [1,2,3];
 end


[~,~,p] = size(data);


dist = zeros(p,1);
for j = 1:p
    point_dist = 0;
    for n_i = flag_type
        x = data(:,n_i,j);
        y = Y(:,n_i);
        point_dist = point_dist + 1- trace(y' * x *x'* y);    
    end
    
    if point_dist >= 0
        point_dist = (point_dist)^(1/2);
    else
        point_dist = 0;
    end

    dist(j) = point_dist;
end