function [Rmle info params details] = synchronizeMLEplus(problem, R0, options)
% Run the synchronizeMLE algorithm on the problem described in the problem
% structure, using the values of kappa1, kappa2 and p in the problem
% structure as first guess of those parameters, then iterate: based on the
% estimation of the rotations, we estimate the noise and from there
% estimate values of kappa1, kappa2 and p (assuming i.i.d. measurements),
% and go again. This algorithm was first described in Nicolas Boumal's PhD
% thesis: see that document for a reference.
%
% See in code for the options structure (optional).
%
% See also: synchronizeMLE langevinmixturefit
%
% Nicolas Boumal, UCLouvain, Oct. 4, 2013.
    
    if ~exist('problem', 'var') || ~isstruct(problem)
        error('The first input parameter (problem) must be a structure. See buildproblem.');
    end

    M = problem.M;
    I = problem.I;
    J = problem.J;
    H = problem.H;
    
    % Force the original noise model to be an i.i.d. model: identical
    % parameters the same for all edges. It's just more convenient, but not
    % exactly necessary, fundamentally.
    problem.kappa1 = mean(problem.kappa1)*ones(M, 1);
    problem.kappa2 = mean(problem.kappa2)*ones(M, 1);
    problem.p = mean(problem.p)*ones(M, 1);
    
    if ~exist('R0', 'var') || isempty(R0)
        start = tic();
        R0 = initialguess(problem);
        synchro_time = toc(start);
    else
        synchro_time = 0;
    end
    
    if ~exist('options', 'var')
        options = struct();
    end
    
    if ~isstruct(options)
        error('The third input parameter (options), if provided, must be a structure.');
    end
    
    % Specify default options here. The user supplied options structure
    % will complement (and possibly overwrite) the default options.
    % This same options structure is also passed to synchronizeMLE, which
    % will pass it to Manopt's trustregions algorithm, so you may use it to
    % specify options for that part too.
    default_options.param_tolerance = 1e-3;
    default_options.verbosity = 1;
    default_options.max_mleplus_iterations = 20;
    % Do not activate the flag below if the algorithm is timed for speed.
    default_options.compute_fitting_errors = false;
    options = mergeOptions(default_options, options);
    
    % We will use this as the first guess in the loop.
    Rmle = R0;
    
    % details is a struct array containing full information about the
    % outer iterations of MLE+. This is not required to run the algorithm:
    % if you feel this is slowing the algorithm down, you may remove
    % everything involved with this structure.
    detail.iter = 0;
    detail.synchro_time = synchro_time;
    detail.Rmle = Rmle;
    detail.params = struct('kappa1', problem.kappa1(1), ...
                           'kappa2', problem.kappa2(1), 'p', problem.p(1));
    detail.fitting_time = 0;
    details(1) = detail;
    detail = struct();

    % Arbitrary limit on the number of outer itations.
    for ITER = 1 : options.max_mleplus_iterations
        detail.iter = ITER;

        % Synchronization happens here, following the maximum likelihood
        % procedure with the given parameters for the noise model.
        start = tic();
        [Rmle info] = synchronizeMLE(problem, Rmle, options);
        detail.synchro_time = toc(start);
        detail.Rmle = Rmle;
        
        % Log the parameter values which were used for this Rmle.
        params.kappa1 = mean(problem.kappa1);
        params.kappa2 = mean(problem.kappa2);
        params.p = mean(problem.p);
        detail.params = params;

        % We will try to estimate the distribution of the noise, with
        % parameters kappa1, kappa2 and p. Look at the error on the
        % measurements "if the Rmle was the Rtrue":
        Ri = Rmle(:, :, I);
        Rj = Rmle(:, :, J);
        Zhat = multiprod(multitransp(Ri), multiprod(H, Rj));

        % Compute some statistics about the quality of fit to the
        % measurements we get. This is something which can be computed in
        % applications too: the true rotations are not needed for this
        % quality metric. This is actually expensive to compute: remove if
        % the algorithm is timed for speed.
        if options.compute_fitting_errors && (n == 2 || n == 3)
            fitting_error_degrees = fittingerrors_degrees(problem, Rmle);
            fprintf('  kappa1 = %g, kappa2 = %g, p = %g,\n',...
                       params.kappa1, params.kappa2, params.p);
            fprintf('  Median data fitting error in degrees: %g.\n', ...
                       median(fitting_error_degrees));
            % hist(fitting_error_degrees);
            % pause;
        end

        % Based on Zhat, our estimation of the noise matrices, estimate the
        % parameters of the noise model:
        start = tic();
        new_params = langevinmixturefit(Zhat, params);
        detail.fitting_time = toc(start);
        
        details(end+1) = detail; %#ok<AGROW>
        detail = struct();
    
        % If the parameters for the noise model changed significantly, set
        % them up in the problem structure and go again; otherwise, stop.
        if parameter_distance(params, new_params) >= options.param_tolerance
            problem = changeproblemweights(problem, ...
                                       new_params.kappa1*ones(M, 1), ...
                                       new_params.kappa2*ones(M, 1), ...
                                       new_params.p*ones(M, 1));
        else
            break;
        end
    end
    
end

% Defines a distance on the space of mixtures of langevins, based on the
% parameters kappa1, kappa2 and p.
function dist = parameter_distance(param1, param2)

    vec = zeros(3, 1);
    vec(1) = log(param1.kappa1) - log(param2.kappa1);
    vec(2) = log(param1.kappa2) - log(param2.kappa2);
    vec(3) = param1.p - param2.p;

    dist = norm(vec, inf);

end
