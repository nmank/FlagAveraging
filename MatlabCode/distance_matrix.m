function [D] = distance_matrix(data, centers, flag_type)
%centers by data matrix
[~,~,m] = size(centers);
[~,~,n] = size(data);

D = zeros(m,n);


for i = 1:m
    for j = 1:n
        D(i,j) = chordal_distance(data(:,:,j), centers(:,:,i), flag_type);
    end
end

