function L = synchrolaplacian(problem)
% Returns the Laplacian of the information weighted synchronization graph
% for the problem of synchronization of rotations described in 'problem'.
%
% See the associate paper:
% http://imaiai.oxfordjournals.org/content/early/2013/09/23/imaiai.iat006.abstract
% N. Boumal, A. Singer, P.-A. Absil and V. D. Blondel,
% Cramér-Rao bounds for synchronization of rotations,
% in Information and Inference, a journal of the IMA, 2013.
%
% See also: synchrocrb infoweights
%
% Nicolas Boumal, UCLouvain, 2013.

    n = problem.n;
    N = problem.N;
    I = problem.I;
    J = problem.J;
    kappa1 = problem.kappa1;
    kappa2 = problem.kappa2;
    p = problem.p;

    w = infoweights(n, kappa1, kappa2, p);
    
    L = laplacian(N, I, J, w);

end
