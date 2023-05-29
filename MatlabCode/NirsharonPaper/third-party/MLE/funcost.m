function [cost store] = funcost(problem, R, store)
%  cost = funcost(problem, R)
%
%  Compute the log-likelihood of an estimator R (nxnxN matrix) for the
%  problem described in the problem structure. See this paper:
%
%  N. Boumal, A. Singer and P.-A. Absil, 2013,
%    Robust estimation of rotations from relative measurements
%    by maximum likelihood,
%  In the proceedings of the 52nd Conference on Decision and Control (CDC).
%
% [cost store] = funcost(problem, R, store)
%
%  The store structure is as described in the documentation of Manopt: it
%  is a structure used as caching mechanism, such that each time a
%  cost-related function is called at R (be it the cost, the gradient or
%  the Hessian), the associated store structure is passed along. This makes
%  it possible to cut down on redundant computations.
% 
% See also: fungrad funhess synchronizeMLE
%
% Nicolas Boumal, UCLouvain, Nov. 20, 2012.

    if ~exist('store', 'var')
        store = struct();
    end
    
    if isfield(store, 'cost')
        cost = store.cost;
        return;
    end
    
    
    %% Extract data from the problem structure.
    n = problem.n;
    M = problem.M;
    H = problem.H;
    I = problem.I;
    J = problem.J;
    p = problem.p;
    c1 = problem.c1;
    c2 = problem.c2;
    kappa1 = problem.kappa1;
    kappa2 = problem.kappa2;

    
    %% Compute the cost
    Ri = R(:, :, I);
    Rj = R(:, :, J);
    hatZ = multiprod(multitransp(Ri), multiprod(H, Rj));
    
    trace_hatZ = multitrace(hatZ);
    
    ell1 =    p  .* exp(kappa1.*(trace_hatZ-n))./c1;
    ell2 = (1-p) .* exp(kappa2.*(trace_hatZ-n))./c2;
    
    f = ell1 + ell2;
    
    cost = -sum(log(f))/M;
    
    
    %% Store some data for the gradient and the Hessian functions.
    store.cost = cost;
    store.hatZ = hatZ;
    store.ell1 = ell1;
    store.ell2 = ell2;
    store.f = f;

end
