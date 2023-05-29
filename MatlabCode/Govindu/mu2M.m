function M=mu2M(m)
w=[m(3,2),m(1,3),m(2,1)];
theta=(norm(w));
t=(m(1:3,4));
M(1:3,1:3)=w2R(w);
if(theta)
    tw=w'*(w*t)/theta/theta;
    tp=t-tw;
    M(:,4)=2*sin(theta/2)/theta*w2R(w/2)*tp+tw;
else
    M(:,4)=t;
end
M(4,:)=[0 0 0 1];