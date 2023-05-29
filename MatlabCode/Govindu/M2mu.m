function m =M2mu(M)
w=R2w(M(1:3,1:3))';
theta=norm(w);
m=crossmat(w);
T=M(1:3,4);
if(theta)
    Tw=w*(w'*T)/theta/theta;
    Tp=T-Tw;
    m(:,4)=w2R(w'/2)'*Tp*theta/2/sin(theta/2)+Tw;
else
    m(:,4)=T;
end
m(4,:)=[0 0 0 0];