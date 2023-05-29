function [dist] = calculate_chordal_distance(data, Y, flag_type, median)

[n,k,p] = size(data);


dist = 0;
for j = 1:p
    point_dist = 0;
    for n_i = flag_type
        x = data(:,n_i,j);
        y = Y(:,n_i);
        point_dist = point_dist + 1- trace(y' * x *x'* y);    
    end
    if median
        point_dist = (point_dist)^(1/2);
    end
    dist = dist + point_dist;
end