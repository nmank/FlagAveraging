function[ A ]= Inverse_Psi_Lambda_Rod(Q, lambda)
% Apply inverse Psi_lambda specifically for SO(d) (to SE(d-1)) by 
% a generalized Rodrigues Formula

n = size(Q,2);
theta = acos(Q(n,n));
if theta == 0  % Q(n,n)==1 => v==0
    A = Q;
elseif theta == pi  %  => norm(v)==pi, outside cutlocus!     (Q(4,4)==-1)
    rand_circle_pt = randn(n-1,1); 
    rand_circle_pt = rand_circle_pt/norm(rand_circle_pt);
    A = [eye(n-1), lambda*rand_circle_pt; zeros(1,n-1), 1];
    warning('lambda is not large enough');
else
    v = Q(1:(n-1),n)*theta/sin(theta);
    Av = [v;1]*[-v',1]-[v;0]*[-v',0]-diag([zeros(1,n-1),1]);

    % Rodrigues exp
    A1 = -Av/theta;
    mu = (eye(n) + sin(theta)*A1+(1-cos(theta))*A1^2)*Q;
    
    % OLD: mu2 = expm(-Av)*Q;   % we can do it slightly faster with Rodrigues %norm(mu-mu2)
    A = [mu(1:(n-1),1:(n-1)),lambda*v; zeros(1,n-1), 1];

end
end



