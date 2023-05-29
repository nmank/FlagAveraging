function [flg_mean] = chordal_gr_mean(data, weights)

[n,k,p] = size(data);


wt_data = zeros(n,k*p);
for j = 1:p
    wt_data(:,(j-1)*(k)+1:j*k) = weights(j)*data(:,:,j);
end

[U, ~, ~] = svd(wt_data);

flg_mean = U(:,1:k);
