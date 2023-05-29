% test_sync_SEk_by_PD_contraction
%
% N.S, April 2016

n = 40;
k = 5;

lambda = 20;
R = 1;

new_data = 1;%-1;

if new_data
%---- synthetic data in SE(k) ------
SEk_array = zeros(k+1,k+1,n);
for l=1:n
    A = rand(k);
    [q, ~] = qr(A);
    % positive determinant
    q(:,k) = det(q)*q(:,k);
    % transational part
    b = R*rand(k,1);
    % embed in the matrix
    SEk_array(1:k,1:k,l) = q;
    SEk_array(1:k,k+1,l) = b;
    SEk_array(k+1,k+1,l) = 1;
 end

%---- construct the similarity matrix, OUTLIERS model ----
%---- These are the "real" measurements ------------------
%---------------------------------------------------------
s = (k+1)*n;  % the size

p = .16;%0.6;  % the probability of non-outliers
m = n*(n-1)/2;    % full graph
non_outliers = floor(p*m); %number of outliers 
y = randsample(m,non_outliers); 

prob_arr = sparse(n,n);
idx = find(~tril(ones(n)));  % indices of upper side blocks
prob_arr(idx(y))=1;          % mark only the relevant, non-outliers

Affin_mat = zeros(s);
confidence_weights = ones(n);

for l=1:n
    for m=(l+1):n
        ind1 = 1+(l-1)*(k+1);
        ind2 = 1+(m-1)*(k+1);
        if prob_arr(l,m)
            SE_measurement = SEk_array(:,:,l)*inverse_SE_k(SEk_array(:,:,m));
        else   % missing data case / outlier
            SE_measurement = zeros(k+1); confidence_weights(l,m)=0; confidence_weights(m,l)=0;% make_random_SE_k(k);
        end
        Affin_mat(ind1:(ind1+k),ind2:(ind2+k))= SE_measurement;
        Affin_mat(ind2:(ind2+k),ind1:(ind1+k)) = inverse_SE_k(SE_measurement);
    end
end
Affin_mat = Affin_mat +eye(s);

%---- calling the functions -----
end

estimations = sync_SEk_by_PD_contraction( triu(Affin_mat), confidence_weights, k, lambda );
current_err = error_calc_SE_k( estimations, SEk_array )




