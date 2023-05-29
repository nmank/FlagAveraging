function [ mapped_data ] = Psi_lambda_MMG(MMGdata, lambda)
% Psi mapping for MAtrix Motion Group: MMG(d,l) -> O(d+l)
% -- MMG is a cell array of size nX3
% -- The output is array of n matrices of order d+l
%
% NS, June 17

n = size(MMGdata,1);
d = size(MMGdata{1,1},1);
l = size(MMGdata{1,2},1);

mapped_data = zeros(d+l,d+l,n);

for j=1:n    
   M = MMGdata{j,3}/lambda; 
   B = [zeros(l,l), M'; -M, zeros(d,d)]; 
   % the mapping
   mapped_data(:,:,j) = expm(B)*blkdiag(MMGdata{j,2},MMGdata{j,1});
end

end

