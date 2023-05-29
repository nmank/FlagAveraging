function problem = buildproblem(n, N, M, I, J, H, kappa1, kappa2, p, A, Ra)
%
% TODO : refactor
%
% Returns a problem structure containing all information pertaining to a
% rotation synchronization problem.
%
% INPUTS:
%
%  n = 2, 3 or 4: work with rotations in SO(n), i.e., n-by-n orthog. matrices
%  N : number of rotations to synchronize
%  M : number of measurements of rotation ratios
%  I, J : length-M vectors of indices between 1 and N
%  H : n-by-n-by-M matrix; Each matrix H(:, :, k) in SO(n) is a measurement
%      of the ratio Ra*Rb', where Ra is the I(k)'s rotation matrix to
%      synchronize and Rb is the J(k)'s rotation matrix to synchronize
%  kappa1,2 : length-M vectors of confidences in the M measurements
%  p : probability that a measurement is not an outlier
%  A : (optional) vector of indices of the anchors of the problem, i.e.,
%      indices of the known rotation matrices; if omitted, replaced by [1].
%  Ra : (optional) nxnx|A| matrix with anchored rotations; if omitted,
%       replaced by the identity matrix eye(n).
% 
% OUTPUTS:
%
%  problem : a structure containing all the given information plus some
%            precomputed data.
%
% Nicolas Boumal, UCLouvain, Aug. 17, 2011
    
    problem = struct();
    
    problem.n = n;
    problem.N = N;
    problem.M = M;
    problem.I = I;
    problem.J = J;
    problem.H = H;
    problem.kappa1 = kappa1;
    problem.kappa2 = kappa2;
    problem.p = p;
    
    % date of birth of the problem structure
    problem.dob = clock();
    
    if exist('A', 'var') && ~isempty(A)
        problem.A = A;
        problem.Ra = Ra;
    else
        problem.A = 1;
        problem.Ra = eye(n);
    end
    
    scaled = true;
    problem.c1 = langevinnormalizations(n, kappa1, scaled);
    problem.c2 = langevinnormalizations(n, kappa2, scaled);
    
    % d(i) is the number of measurements involving rotation i, i.e., it is
    % the degree of node i in the measurement graph.
    problem.d = hist([I ; J], 1:N).';
    
    if ~isconnected(problem.N, problem.I, problem.J)
        warning('synch:disconnected', 'The measurement graph is disconnected!');
        problem.disconnected = true;
    else
        problem.disconnected = false;
    end
    
    [maskI maskJ] = computemasks(N, I, J);
    problem.maskI = maskI;
    problem.maskJ = maskJ;
    
end
