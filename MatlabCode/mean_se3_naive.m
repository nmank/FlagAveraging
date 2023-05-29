% the most naive thing we can do : average the matrices and project onto
% SE(3)
function [Mu] = mean_se3_naive(Ts, Weights)

N = length(Ts);
Mu = zeros(4);
for i=1:N
    Ti = Ts{i};
    Mu = Mu + Weights(i).*Ti;
end

Mu = Mu./N;

% now project
Mu(1:3,1:3) = project_onto_so3(Mu(1:3,1:3));

end
