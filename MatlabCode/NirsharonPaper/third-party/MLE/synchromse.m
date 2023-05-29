function mse = synchromse(problem, Rtrue, Rest)
% Computes the mean squared error criterion for the Rest estimator of the
% Rtrue rotations in the synchronization problem described by the problem
% structure.
%
% Only supports anchored synchronization for now.
% See soregister for anchor-free MSE.
%
% Nicolas Boumal, UCLouvain, Jan. 18, 2013.

    n = problem.n;
    N = problem.N;
    A = problem.A;
    Ra = problem.Ra;
    manifold = anchoredrotationsfactory(n, N, A, Ra);
    
    mse = manifold.dist(Rtrue, Rest)^2 / (N-max(1, numel(A)));

end
