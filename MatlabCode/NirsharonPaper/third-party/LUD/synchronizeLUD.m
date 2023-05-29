function R = synchronizeLUD(problem)

    n = problem.n;
    N = problem.N;
    M = problem.M;
    H = problem.H;
    I = problem.I;
    J = problem.J;
    d = problem.d;
    A = problem.A;
    Ra = problem.Ra;
    
    if any(d <= 0)
        error('All nodes must have positive degree.');
    end
    
    % Synchronization matrix, made of nxn blocks corresponding to the
    % weighted measurements.
    W1 = spalloc(n*N, n*N, n^2*(2*M+N));
    
    % Weight vector: D(i) is the sum of the kappa1 weights of the edges
    % adjacent to node i. There is a weight of 1 for the identity matrix
    % which comes from the self-loop measurement.
    D = ones(N, 1);
    
    % TODO: compute list of indices and populate sparse matrix all at once.
    for k = 1 : M
        
        i = I(k);
        j = J(k);
        weight = 1; %problem.kappa1(k);
        
        W1( (i-1)*n + (1:n), (j-1)*n + (1:n) ) = weight*H(:, :, k);
        
        D(i) = D(i) + weight;
        D(j) = D(j) + weight;
    end
    
    W1 = W1 + W1';
    W1(1:n*N+1:end) = 1;
    
    % Denoise the synchronization matrix with the LUD algorithm-
    G0 = eye(n*N);
    opts.delta_mu_l = 0.1;
    opts.delta_mu_u = 10;
    opts.tol = 1e-4;
    % opts.max_nev = 6;
    opts.record = 1;
    opts.maxit = 500;
    weights = sparse([(1:N)' ; I ; J], ...
                     [(1:N)' ; J ; I], ones(N+2*M, 1), N, N, N+2*M);
    [W1, W, y, out] = syncL1(N, n, G0, W1, weights, opts);
    dd = repmat(D, 1, n).';
    D1 = spdiags(dd(:), 0, n*N, n*N);
    
    [X E] = eigs(W1, D1, n); %#ok<NASGU>
    
    R1 = zeros(n, n, N);
    R2 = zeros(n, n, N);
    J = diag([ones(n-1, 1); -1]);
    for i = 1 : N
        Xi = X( (i-1)*n + (1:n) , :);
%         Xi = Xi';
        R1(:, :, i) = soregister(Xi);
        R2(:, :, i) = soregister(Xi*J);
    end
    
    % pick the most likely estimator
    % The J-operation is not tied to an invariance of the problem but
    % rather to the fact that the eigenvector method returns n eigenvectors
    % which can all be in one direction or another, independently from one
    % another. We may only choose to reverse one column (the last one for
    % example), and do so if the likelihood is improved by it.
    l1 = funcost(problem, R1);
    l2 = funcost(problem, R2);
    if l1 < l2
        R = R1;
    else
        R = R2;
    end
    
    Q = soregister(Ra, R(:, :, A));
    R = multiprod(R, Q);
    R(:, :, A) = Ra;

end
