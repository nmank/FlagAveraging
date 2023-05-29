% quaternion rotation averaging + translation averaging
function [Mu] = mean_se3_quat_tra(Ts, Weights)

N = length(Ts);
if (~exist('Weights','var'))
    Weights = ones(N,1);
end

Q = zeros(N, 4);
T = zeros(N, 3);
for i=1:N
    Ti = Ts{i};
    Q(i,:) = dcm2quat(Ti(1:3,1:3));
    T(i,:) = Ti(1:3, 4)';
end

q_mean = wavg_quaternion_markley(Q, Weights);
t_mean = mean(T,1);

Mu = [quat2dcm(q_mean') t_mean'; [0,0,0,1]];

end