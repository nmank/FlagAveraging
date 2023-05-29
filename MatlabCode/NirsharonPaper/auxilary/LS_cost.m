function[cost] = LS_cost(estimation, A, W)
% the cost function in use for the LS optimization
% estimation is d+1 X d+1 X n;

n = size(estimation,3);
d = size(estimation,1)-1;
X.R = zeros(d,d,n);
X.t = zeros(d,n);
for j=1:n
        X.R(:,:,j) = estimation(1:d,1:d,j);
        X.t(:,j) = estimation(1:d,d+1,j);
end

cost = 0;
for i=1:n
    for j=(i+1):n
        if W(i,j)>0
            ind1 = 1+(i-1)*(d+1);
            ind2 = 1+(j-1)*(d+1);
            b_ij = A(ind1:(ind1+d-1),ind2+d);
            mu_ij = A(ind1:(ind1+d-1),ind2:(ind2+d-1));
            term1 = .5*(norm(mu_ij*X.R(:,:,j)-X.R(:,:,i),'fro')^2);
            % second term
            term2 = (1/d)*.5*norm(X.R(:,:,i)*X.R(:,:,j)'*X.t(:,j)-(X.t(:,i)-b_ij))^2;
            cost = cost + term1 + term2;
        end
    end
end