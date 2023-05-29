function [Rmle info] = synchronizeMLE(problem, R0, options)
% Solves the synchronization problem using a maximum likelihood approach.
%
% function [Rmle info] = synchronizeMLE(problem, R0, options)
%
% SynchronizeMLE uses optimization on a manifold via Manopt. The Manopt
% toolbox can be downloaded from http://www.manopt.org. You need to install
% it before using this function. The method used here is described in:
%
% N. Boumal, A. Singer and P.-A. Absil, 2013,
%   Robust estimation of rotations from relative measurements
%   by maximum likelihood,
% in the proceedings of the 52nd Conference on Decision and Control (CDC).
%
% A synchronization of rotations problem instance must be described using a
% problem structure (first input argument). To create this structure from
% your problem data, use the buildproblem function. See also main.m for an
% example.
%
% R0 is an initial guess for the rotations. If none is provided (or [] is
% provided) then the initialguess function will be used to generate one.
%
% The options structure will be passed to Manopt's trustregions algorithm.
% See the help for that solver for details.
%
% See also: buildproblem initialguess
%
% Nicolas Boumal, UCLouvain, Jan. 16, 2013.
    
    if ~exist('problem', 'var') || ~isstruct(problem)
        error('The first input parameter (problem) must be a structure. See buildproblem.');
    end

    % Extract the problem parameters and anchors
    n = problem.n;
    N = problem.N;
    A = problem.A;
    Ra = problem.Ra;

    if ~exist('R0', 'var') || isempty(R0)
        R0 = initialguess(problem);
    end
    
    if ~exist('options', 'var')
        options = struct();
    end
    
    if ~isstruct(options)
        error('The third input parameter (options), if provided, must be a structure.');
    end
    
    % Specify default options here. The user supplied options structure
    % will complement (and possibly overwrite) the default options.
    default_options.maxinner = 100;
    default_options.tolgradnorm = 1e-6;
    default_options.debug = false;
    default_options.verbosity = 1;
    options = mergeOptions(default_options, options);
    
    % Obtain a Manopt description of the manifold: product of rotation
    % groups, with anchors indexed in A and given by Ra.
    manifold = anchoredrotationsfactory(n, N, A, Ra);
    optiproblem.M = manifold;

    % Specify the cost function and its derivatives. Notice that we use the
    % store caching capability of Manopt to cut down on redundant
    % computations.
    optiproblem.cost = @(R, store) funcost(problem, R, store);
    optiproblem.grad = @(R, store) fungrad(problem, R, store);
    optiproblem.hess = @(R, Omega, store) funhess(problem, R, Omega, store);
    
    % For debugging purposes, it is nice to check numerically that the
    % gradient and the Hessian are compatible with the cost function.
    % figure(1); checkgradient(optiproblem); pause;
    % figure(2); checkhessian(optiproblem); pause;
    
    % The magic happens here: call the Riemannian trust-regions
    % optimization algorithm to maximize the likelihood (actually, to
    % minimize the opposite of the log-likelihood, normalized by the number
    % of measurements).
    [Rmle, ~, info] = trustregions(optiproblem, R0, options);

end
