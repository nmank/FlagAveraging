function scores = pagerankscores(problem, R)
% scores = pagerankscores(problem, R)
%
% Estimate a PageRank style score for the quality of each rotation
% estimator R(:,:,i) based on consistency with the measurements.
%
% Nicolas Boumal, UCLouvain, Sept. 30, 2011.
% Major update: Feb. 11, 2013.

    if problem.disconnected
        warning('synch:pagerankdisconnected', ...
               ['The measurement graph is disconnected, ' ...
                'hence the PageRank matrix is not irreducible, ' ...
                'and the PageRank scores are not well defined.']);
    end

    %% Extract data from problem structure.
    n = problem.n;
    N = problem.N;
    M = problem.M;
    H = problem.H;
    I = problem.I;
    J = problem.J;
    p = problem.p;
    c1 = problem.c1;
    kappa1 = problem.kappa1;

    Ri = R(:, :, I);
    Rj = R(:, :, J);
    hatZ = multiprod(multitransp(Ri), multiprod(H, Rj));
    
    trace_hatZ = multitrace(hatZ);
    
    scoremeasures = exp(kappa1.*(trace_hatZ-n))./c1;
    
    scores = ones(N, 1);
    A = sparse([I;J], [J;I], ...
               [scoremeasures.*sqrt(scores(J)) ; ...
                scoremeasures.*sqrt(scores(I))], ...
                N, N, 2*M);
    S = full(sum(A, 2));
    A = spdiags(1./S, 0, N, N) * A;
    % Look for an eigenvector associated to the eigenvalue 1.
    % We tell eigs to look for an eigenvalue close to 1.01 because the
    % algorithm doesn't like it too much if the actual value of the
    % eigenvalue is specified.
    [scores ~] = eigs(A', 1, 1.01);
    scores = scores*(N/sum(scores));
    
end
