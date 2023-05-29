function [ estimations ] = sync_flag( Affinity_mat, confidence_weights, k, lambda)

n_its = 1000;

% the affinity matrix
s = size(Affinity_mat,1);
n = size(confidence_weights,1);
ord = k+1;
if n*(ord)~=s
    error('wrong matrices sizes');
end

% initialize SO guesses
so_guesses = zeros(ord,ord,n);
for i = 1:n
    rand_mat = rand(k+1);
    [u,~,v] = svd(rand_mat);
    so_guesses(:,:,i) = u*diag([det(u*v');1;1;1])*v';
end

%make SO affinity matrix %from nir sharon
SO_affinity = zeros(s);
for l=1:n
    for j=(l+1):n
        if confidence_weights(l,j)>0
            ind1 = 1+(l-1)*(ord);
            ind2 = 1+(j-1)*(ord);
            mu = Affinity_mat(ind1:(ind1+k-1),ind2:(ind2+k-1));
            b  = Affinity_mat(ind1:(ind1+k-1),ind2+k);
            SO_affinity(ind1:(ind1+k),ind2:(ind2+k)) = se_to_so(mu,b,lambda);         
        end
    end
end


for itr=1:n_its

    % construct the affinity matrix in SO(k+1) using Psi_lambda by tranform
    % each block
    for l=1:n
        %init so list
        flag_points = zeros(k+1,k,n);
        for j=1:n
            if confidence_weights(l,j)>0

                ind1 = 1+(l-1)*(k);
                ind2 = 1+(j-1)*(k);

                %multiply by inverse of guess
                so_mult = so_guesses(:,:,j)'*SO_affinity(ind1:(ind1+k),ind2:(ind2+k));

                %map to flag
                flag_points(:,:,j) = so_to_flag(so_mult);
            end
        end

        %compute average in flag
        avg_flag_point = chordal_flag_mean(flag_points, confidence_weights(l,:));
        
        %map to SO
        so_guesses(:,:,l) = flag_to_so(avg_flag_point);
    end
    %check convergence

end

estimations = zeros(k+1,k+1,n);
for l=1:n
    [mu,b] = so_to_se(so_guesses(:,:,l), lambda);

    A = zeros(k+1);
    A(k+1,k+1) = 1;
    A(1:k,k+1) = b;
    A(1:k,1:k) = mu;
    
    estimations(:,:,l) = A;
end


