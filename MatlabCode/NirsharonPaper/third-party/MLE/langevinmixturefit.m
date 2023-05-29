function params = langevinmixturefit(Z, params)
% The input Z is a 3D matrix of size nxnxM such that each slice Z(:, :, k)
% is a rotation matrix. This function attempts to fit a mixture of Langevin
% distribution to these rotation matrices. The parameters are kappa1,
% kappa2 and p. 
% The input params is a structure containing an initial guess for the model
% and the output is the best fit we could reach, starting from the initial
% guess. This is probably fairly sensitive to the initial guess, hence we
% do not build a default initial guess. This being said, setting
% params = struct('kappa1', 10, 'kappa2', 0, 'p', 1);
% might be a good generic start: it assumes no outliers in the data.
%
% This paper describes the mixture of Langevin distribution:
% N. Boumal, A. Singer and P.-A. Absil, 2013,
%   Robust estimation of rotations from relative measurements
%   by maximum likelihood,
% in the proceedings of the 52nd Conference on Decision and Control (CDC).
%
% See also Nicolas Boumal's PhD thesis for a description of the algorithm
% used in this function.
%
% Nicolas Boumal, UCLouvain, Oct. 7, 2013.

    n = size(Z, 1);
    assert(size(Z, 2) == n);
    
    % Change of variables, see explanation below.
    % The vectors x are of the form [kappa1, kappa2, p] or of the form
    % [gamma1, gamma2, q]. These scalings are inverse to each other for q
    % restricted to one period.
    scale_forward  = @(x) [log(x(1)); log(x(2)); acos(2*x(3)-1)];
    scale_backward = @(x) [exp(x(1)); exp(x(2)); .5*(cos(x(3))+1)];
    
    traceZ = multitrace(Z);
    
    % For p = 0 or p = 1, the change of variable on p makes the derivative
    % wrt p structurally zero, so in order to allow for p to move away from
    % those values if it needs to, we force the initial value of p to be
    % bounded away from them. If it was optimal anyway, it will probably
    % move back there...
    params.p = max(.01, min(.99, params.p));
    % Similarly, a zero value for either kappa concentration results in
    % infinite or nan values in the cost and gradient, which we'd rather
    % avoid.
    params.kappa1 = max(1e-6, min(1e6, params.kappa1));
    params.kappa2 = max(1e-6, min(1e6, params.kappa2));
    
    % Build an initial guess for the scaled parameter fit cost function,
    % then build a manopt problem structure to pass on to the optimization
    % stage.
    x0 = scale_forward([params.kappa1; params.kappa2; params.p]);
    problem.M = euclideanfactory(3, 1);
    problem.costgrad = @scaled_parameter_cost;
    options.verbosity = 0;
    optim_x = conjugategradient(problem, x0, options);
    
    % Scale back the optimal point to obtain the estimate of the parameters
    optim_params = scale_backward(optim_x);
    kappa1 = optim_params(1);
    kappa2 = optim_params(2);
    p      = optim_params(3);
    
    % Normally, we let the more concentrated Langevin be the first one, so
    % if the optimization resulted in kappa1 < kappa2, exchange their
    % roles, which entails changing p for 1-p also.
    if kappa1 < kappa2
        temp = kappa1;
        kappa1 = kappa2;
        kappa2 = temp;
        p = 1-p;
    end
    
    params.kappa1 = kappa1;
    params.kappa2 = kappa2;
    params.p      = p;
    
    
    % What we want to minimize is parameter_cost (blow), but the parameters
    % kappa1 and kappa2 naturally scale logarithmically, and p scales
    % linearly between 0 and 1. Hence, to keep things balanced, we optimize
    % the function below, where instead of kappa1 and kappa2 the inputs are
    % gamma1 and gamma2, related to the former ones by kappa = exp(gamma).
    % Furthemore, we optimize for q which is related to p via
    % 2p = cos(q)+1. This way, the constraint that p remains between 0 and
    % 1 is automatically satisfied.
    % It's just a function composition, and we follow the composition rule
    % to adapt the derivatives, that's all.
    function [val grad] = scaled_parameter_cost(scaled_x)
        
        x = scale_backward(scaled_x);
        
        [val grad] = parameter_cost(x);
        
        % Composition rule for the derivative of the scaled cost in terms
        % of the derivative of the base cost.
        grad(1) = grad(1)*x(1);
        grad(2) = grad(2)*x(2);
        grad(3) = grad(3)*(-.5*sin(scaled_x(3)));
        
    end

    % Compute the likelihood of the parameters kappa1, kappa2 and p for the
    % given data traceZ, but multiplied by -1/M (M being the number of
    % matrices in Z) because (1) we want to maximize the function but most
    % optimization packages minimize, and (2) dividing by M keeps the scale
    % of the cost function insensitive to the size of the data.
    function [val grad] = parameter_cost(x)

        kappa1 = x(1);
        kappa2 = x(2);
        p      = x(3);
        
        c1 = langevinnormalization(n, kappa1, true);
        c2 = langevinnormalization(n, kappa2, true);
        
        ell1 = exp(kappa1*(traceZ - n))/c1;
        ell2 = exp(kappa2*(traceZ - n))/c2;
        
        f = p*ell1 + (1-p)*ell2;

        val = -mean(log(f));
        
        % Expected value of each entry in traceZ if the matrices in Z where
        % distributed following a single Langevin with concentration either
        % kappa1 or kappa2.
        expected_trace1 = n*langevinphasetransition(n, kappa1);
        expected_trace2 = n*langevinphasetransition(n, kappa2);
        
        % derivative of ell1 and ell2 w.r.t. kappa1 or kappa2
        dell1 = (traceZ-expected_trace1).*ell1;
        dell2 = (traceZ-expected_trace2).*ell2;
        
        grad = zeros(size(x));
        grad(1) = -p*mean(dell1./f);
        grad(2) = -(1-p)*mean(dell2./f);
        grad(3) = -mean((ell1-ell2)./f);
        
        % When p = 1, numerics get a little tense ... as a workaround
        % for lack of a better way of handling all inf/nan situations,
        % here is a correction code for p = 1.
        % We should do the same for p = 0, symmetrically, but p = 0 is
        % usually not an interesting point.
        if p == 1
            logf = kappa1*(traceZ - n) - log(c1);
            val = -mean(logf);
            dell1_over_f = traceZ-expected_trace1;
            ell2_over_ell1 = exp((kappa2-kappa1)*(traceZ - n))*(c1/c2);
            grad(1) = -mean(dell1_over_f);
            grad(2) = 0;
            grad(3) = -mean(1-ell2_over_ell1);
        end
        
        if p == 0
            warning('langfit:pzero', ...
                    'This code is not intended to work with p = 0 ...');
        end

    end

end
