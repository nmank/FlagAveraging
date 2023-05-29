function problem = changeproblemweights(problem, kappa1, kappa2, p)
% Build a synchronization problem structure equivalent to the one given as
% input but with different weights for the measurements.

    n = problem.n;
    N = problem.N;
    M = problem.M;
    I = problem.I;
    J = problem.J;
    H = problem.H;
    A = problem.A;
    Ra = problem.Ra;
    problem = buildproblem(n, N, M, I, J, H, kappa1, kappa2, p, A, Ra);

end
