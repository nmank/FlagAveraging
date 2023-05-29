function [ res ] = LS_svd(A,b)

% Least squares based on svd

[u, s, v] = svd(A,0);

sd = diag(s);
thrhd = 1e-14;
locations = (abs(sd)>thrhd);
inv_s = zeros(numel(sd),1);
inv_s(locations) = sd(locations).^(-1);

D = diag(inv_s);
if size(v,2)>size(D,1)
    diff = size(v,2)-size(D,1);
    D = [D; zeros(diff, size(D,2))];
end
pinv_A = v*D*u';
res = pinv_A*b;


end

