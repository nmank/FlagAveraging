% test_sync_SO_by_maximum_likeliwood
%
% N.S, April 2016

n = 80;
k = 4;

%---- synthetic data in SO(k) ------
SOk_array = zeros(k,k,n);
for l=1:n
    A = rand(k);
    [q, ~] = qr(A);
    % positive determinant
    q(:,k) = det(q)*q(:,k);
    SOk_array(:,:,l) = q;
end

%---- construct the similarity matrix, OUTLIERS model ----
s = k*n;  % the size

p = .7;  % the probability of non-outliers
m = n*(n-1)/2;    % full graph
non_outliers = floor(p*m); %number of outliers 
y = randsample(m,non_outliers); 

prob_arr = zeros(n);
idx = find(~tril(ones(n)));  % indices of upper side matrix
prob_arr(idx(y))=1;          % mark only the relevants

Affin_mat = zeros(s,s);
for l=1:n
    for m=(l+1):n
        ind1 = 1+(l-1)*k;
        ind2 = 1+(m-1)*k;
        if prob_arr(l,m)
            Affin_mat(ind1:(ind1+k-1),ind2:(ind2+k-1))= SOk_array(:,:,l)*SOk_array(:,:,m)';
        else
            [q, ~] = qr(rand(k));
            % positive determinant
            q(:,k) = det(q)*q(:,k);
            Affin_mat(ind1:(ind1+k-1),ind2:(ind2+k-1)) = q;
        end
    end
end
Affin_mat = Affin_mat + Affin_mat';
c = 1;%p;
Affin_mat = Affin_mat+c*eye(n*k);

confidence_weights = ones(n);

%---- calling the function -----
estimations = sync_SO_by_maximum_likeliwood(Affin_mat, confidence_weights, k );

error_mle = error_calc_SO_k(estimations, SOk_array)

Reig = Eigenvectors_Sync_SOk( Affin_mat, confidence_weights, k );

error_eig = error_calc_SO_k(Reig, SOk_array)



