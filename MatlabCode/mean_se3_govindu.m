% lie algebraic motion averaging
% re-implementation of Govindu's paper
function [Mu] = mean_se3_govindu(Ts, Weights, thresh)

N = length(Ts);
if (~exist('Weights','var'))
    Weights = ones(N,1);
end

if (~exist('thresh','var'))
    thresh = 0.01;
end

Mu = eye(4);
iter = 0;
while (iter<20) % max of 5 iterations (should converge)
    deltaT = zeros(4);
    for i=1:N
        DeltaT = inv(Mu) * Ts{i};
        deltaT = deltaT + Weights(i) * M2mu(DeltaT);
    end
    deltaT = deltaT./N ;
    deltaMu = mu2M(deltaT);
    Mu = Mu * deltaMu;
    update = norm(deltaT);
    if (update<thresh)
       break;
    end
    iter = iter + 1;
end

end