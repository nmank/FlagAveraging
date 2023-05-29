function [grad store] = fungrad(problem, R, store)
%  grad = fungrad(problem, R)
% [grad store] = fungrad(problem, R, store)
%
% Computes the gradient of funcost at R.
%
% See also: funcost funhess
%
% Nicolas Boumal, UCLouvain, Nov. 20, 2012.

    if ~exist('store', 'var')
        store = struct();
    end
    
    if isfield(store, 'grad')
        grad = store.grad;
        return;
    end
    
    
    %% Extract data from problem structure.
    n = problem.n;
    M = problem.M;
    A = problem.A;
    maskI = problem.maskI;
    maskJ = problem.maskJ;
    kappa1 = problem.kappa1;
    kappa2 = problem.kappa2;
    
    
    %% Enforce the availability of data that the cost function is in
    %  charge of producing.
    if ~isfield(store, 'hatZ') || ~isfield(store, 'ell1') || ...
       ~isfield(store, 'ell2') || ~isfield(store, 'f')
        [~, store] = funcost(problem, R, store);
    end
    hatZ = store.hatZ;
    ell1 = store.ell1;
    ell2 = store.ell2;
    f = store.f;
    

    %% Compute the gradient
    df = kappa1.*ell1 + kappa2.*ell2;
    g = df ./ f;
    
    g_hatZ = multiscale(g, hatZ);
    
    grad = zeros(size(R));
    
    % We write the code for the gradient in this way to avoid looping over
    % M elements, seen as M may be of order N^2 and Matlab doesn't like big
    % loops. An equivalent but much slower code is given below, for
    % readability.
    for k1 = 1 : n
        for k2 = 1 : n
            grad(k1, k2, :) = maskI * (squeeze(g_hatZ(k1, k2, :))) ...
                            - maskJ * (squeeze(g_hatZ(k1, k2, :)));
        end
    end
    
    % for k = 1 : M
    %     i = I(k);
    %     j = J(k);
    %     grad(:, :, i) = grad(:, :, i) + g_hatZ(:, :, k);
    %     grad(:, :, j) = grad(:, :, j) - g_hatZ(:, :, k);
    % end
    
    
    grad = -grad / M;

    
    %% Project the resulting vector from the ambient space to the tangent
    %  space of P_A. The vector A contains the indices of anchored
    %  rotations: they are fixed, hence their gradient component is zero.
    grad = .5*(grad - multitransp(grad));
    grad(:, :, A) = 0;
    
    
    %% Store some data for the Hessian function.
    store.grad = grad;
    store.g = g;

end
