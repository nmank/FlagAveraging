function[cost] = JustCostFunc(X, A, W, n, d)
cost = 0;
for i=1:n
    for j=(i+1):n
        if W(i,j)>0
            ind1 = 1+(i-1)*(d+1);
            ind2 = 1+(j-1)*(d+1);
            b_ij = A(ind1:(ind1+d-1),ind2+d);
            mu_ij = A(ind1:(ind1+d-1),ind2:(ind2+d-1));
            term1 = .5*(norm(mu_ij*X(1:d,1:d,j)-X(1:d,1:d,i),'fro')^2);
            % second term
            term2 = .5*norm(X(1:d,1:d,i)*X(1:d,1:d,j)'*X(1:d,d+1,j)-(X(1:d,d+1,i)-b_ij))^2;
            cost = cost + term1 + term2;
        end
    end
end