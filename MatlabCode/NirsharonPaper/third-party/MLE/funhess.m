function [hess store] = funhess(problem, R, Omega, store)
%  hess = funhess(problem, R, Omega)
% [hess store] = funhess(problem, R, Omega, store)
%
% Computes the Hessian of funcost at R along R*Omega (Omega contains
% skew-symmetric matrices).
%
% See also: funcost fungrad
%
% Nicolas Boumal, UCLouvain, Nov. 22, 2012.

    if ~exist('store', 'var')
        store = struct();
    end
    
    
    %% Extract data from problem structure.
    n = problem.n;
    M = problem.M;
    A = problem.A;
    I = problem.I;
    J = problem.J;
    maskI = problem.maskI;
    maskJ = problem.maskJ;
    kappa1 = problem.kappa1;
    kappa2 = problem.kappa2;
    

    %% Enforce the availability of data that the cost and the gradient
    %  functions are in charge of producing.
    if ~isfield(store, 'hatZ') || ~isfield(store, 'ell1') || ...
       ~isfield(store, 'ell2') || ~isfield(store, 'f') || ...
       ~isfield(store, 'g') || ~isfield(store, 'grad')
        [~, store] = funcost(problem, R, store);
        [~, store] = fungrad(problem, R, store);
    end
    grad = store.grad;
    hatZ = store.hatZ;
    ell1 = store.ell1;
    ell2 = store.ell2;
    f = store.f;
    g = store.g;
    
    
    %% Compute the Hessian
    Omegai = Omega(:, :, I);
    Omegaj = Omega(:, :, J);
    hatOmega = multiprod(hatZ, Omegaj) - multiprod(Omegai, hatZ);

    trace_hatOmega = multitrace(hatOmega);
    
    d2f = kappa1.^2 .* ell1 + kappa2.^2 .* ell2;
    dg = d2f ./ f - g.^2;
    dgtrom = dg .* trace_hatOmega;
    
    hess = zeros(size(R));
    
    elems = multiscale(dgtrom, hatZ) + multiscale(g, hatOmega);
    
    % We write the code for the Hessian in this way to avoid looping over
    % M elements, seen as M may be of order N^2 and Matlab doesn't like big
    % loops. An equivalent but much slower code is given below, for
    % readability.
    for k1 = 1 : n
        for k2 = 1 : n
            hess(k1, k2, :) = maskI * (squeeze(elems(k1, k2, :))) ...
                            - maskJ * (squeeze(elems(k1, k2, :)));
        end
    end
    
    % for k = 1 : M
    %     i = I(k);
    %     j = J(k);
    %     hess(:, :, i) = hess(:, :, i) + elems(:, :, k);
    %     hess(:, :, j) = hess(:, :, j) - elems(:, :, k);
    % end
    
    hess = hess + multiprod(Omega, -M*grad);
    
    hess = -hess / M;
    
    
    %% Project the resulting vector from the ambient space to the tangent
    %  space of P_A. The vector A contains the indices of anchored
    %  rotations: they are fixed, hence their Hessian component is zero.
    hess = multiskew(hess);
    hess(:, :, A) = 0;
    
end
