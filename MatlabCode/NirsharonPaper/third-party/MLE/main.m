% Test file for synchronization of rotations using the maximum likelihood
% algorithm SynchronizeMLE described in
%
% N. Boumal, A. Singer and P.-A. Absil, 2013,
%   Robust estimation of rotations from relative measurements
%   by maximum likelihood,
% in the proceedings of the 52nd Conference on Decision and Control (CDC).
%
% Most of the code is dedicated to producing a random problem instance. If
% you only want to execute the algorithm on your own data, it is helpful to
% read through this script to see how the data must be presented to the
% algorithm, but you will not need that much code of course.
%
% The Cram�r-Rao bounds are described in this reference:
% http://imaiai.oxfordjournals.org/content/early/2013/09/23/imaiai.iat006.abstract
% N. Boumal, A. Singer, P.-A. Absil and V. D. Blondel,
%  Cram�r-Rao bounds for synchronization of rotations,
% in Information and Inference, a journal of the IMA, 2013.
%
% See also: buildproblem synchronizeMLE synchronizeMLEplus
%
% Code: Nicolas Boumal, UCLouvain, 2013.
% Contact: nicolasboumal@gmail.com

clear all;
close all;
clc;

if exist('manopt_version', 'file') ~= 2
    cd 'Manopt_1.0.4';
    importmanopt;
    cd ..;
    warning('manopt:version', ['Using Manopt 1.0.4. This is probably not ' ...
             'the most up-to-date version of Manopt. Please go to ' ...
             'http://www.manopt.org to obtain the latest version.\n\n' ...
             'Manopt 1.0.4 was added to your Matlab path.\n\n']);
end


fprintf('Generating a problem instance... ');

% Synchronize N rotations in SO(n). Rtrue contains the true rotations we
% are looking for. Of course, these are not known for a real problem
% instance.
n = 3;
N = 100;
Rtrue = randrot(n, N);

% Rotations with indices in A are anchored. Ra contains the rotations of
% these anchors. That is: Ra(:, :, k) contains the nxn rotation matrix
% associated to the node A(k), which is anchored. If there are no anchors,
% simply let A = [1] and Ra = eye(n), that is: artificially fix any of the
% rotations to an arbitrary value.
m = 1;
A = 1:m;
Ra = Rtrue(:, :, A);

% Create a representation of the manifold P_A: a product of rotation groups
% with the nodes indexed by A (the anchors) fixed to their known value.
manifold = anchoredrotationsfactory(n, N, A, Ra);

% For this test problem instance, we generate a random Erdos-Renyi graph.
% For each edge in the graph, we will have a measurement of the relative
% rotation between the adjacent nodes. The data is presented this way:
% I, J are two column vectors of length M, where M is the number of edges.
% There is an edge between nodes I(k) and J(k) for k = 1 : M.
% The graph is symmetric, so that we only define each edge once. That is:
% if the matrix [I J] contains the row [a b], then it does not contain the
% row [b a]. ERp is the edge density in the Erdos-Renyi graph.
% For a complete graph, you may use: [I J] = find(triu(ones(N), 1));
ERp = 0.75;
[I J] = erdosrenyi(N, ERp);
M = length(I);

% Pick noise parameters (see the mixture of Langevin distribution described
% in our CDC paper) and generate noise (random rotation matrices) according
% to these parameters. The measurements are stored in H, a 3D matrix such
% that each slice H(:, :, k) is an nxn rotation matrix which corresponds to
% a measurement of the relative rotation Ri Rj^T,
% with Ri = Rtrue(:, :, I(k)) and Rj = Rtrue(:, :, J(k)).
% The measurement H(:, :, k) is distributed around the real relative
% rotation with parameters kappa1(k), kappa2(k) and p(k), where kappa1,
% kappa2 and p are vectors of length M.
% Of course, for a real problem instance, you just need to obtain H from
% your data directly.
kappa1 = 8.0*ones(M, 1);
kappa2 = 0.0*ones(M, 1);
p = 0.70*ones(M, 1);
Z = randlangevinmixture(n, kappa1, kappa2, p);
Htrue = multiprod(Rtrue(:, :, I), multitransp(Rtrue(:, :, J)));
H = multiprod(Z, Htrue);

% Put all the data together in a structure which describes the
% synchronization problem. Here, kappa1, kappa2 and p need not be the same
% as those used in generating the measurements. In fact, for real
% applications, the true values for these noise parameters are unknown. You
% must then guess these parameters, based on prior information you have
% about the measurements (how accurate you think they are, how many
% outliers you expect there to be). If you are not sure, set them to a
% reasonable value and use MLE+: our MLE algorithm which will iteratively
% estimate the rotations then the noise parameters, then the rotations,
% etc. For MLE+, kappa1, kappa2 and p need to be constant vectors.
% Otherwise, only their average is taken into account. If you really don't
% know, try with kappa1 = 10, kappa2 = 0 and p = 0.9. This means you expect
% 10% of outliers and 90% of measurements with roughly 20 degrees errors.
synchroproblem = buildproblem(n, N, M, I, J, H, kappa1, kappa2, p, A, Ra);

fprintf('done.\n');

% This will compute the initial estimator, based on eigenvector
% synchronization. It's a bit different from synchronizeEIG, in that it
% takes anchors into account.
fprintf('Computing the initial guess... ');
R0   = initialguess(synchroproblem);
fprintf('done.\n');

% This will use the initial guess to obtain the MLE estimator, by
% Riemannian trust-regions optimization. You can also pass it an options
% structure as third argument. See code for help on the available options.
options.verbosity = 0;
fprintf('Computing the MLE estimator... ');
Rmle = synchronizeMLE(synchroproblem, R0, options);
fprintf('done.\n');

% In real applications, chances are you don't know the parameters kappa1,
% kappa2 and p for sure. Then, you can use MLE+ instead. This algorithm
% alternatively estimates the rotations and the noise parameters,
% iteratively. See Nicolas Boumal's thesis for details.
% It will use the mean values of kappa1, kappa2 and p given in the
% synchroproblem structure as initial guess for these parameters. All
% measurements are assumed to have the same noise distribution (i.i.d.).
% In this test file, the true parameters are used as initial guess, which
% is not very impressive of course. With real data for which the noise
% distribution is unknown, this algorithm should prove more interesting.
fprintf('Computing the MLE+ estimator... ');
[Rmleplus info params details] = ...
                           synchronizeMLEplus(synchroproblem, R0, options);
fprintf('done.\n');

% Display some statistics
fprintf('Mean squared errors:\n');
fprintf(' MSE R0:    %g\n', synchromse(synchroproblem, Rtrue, R0));
fprintf(' MSE Rmle:  %g\n', synchromse(synchroproblem, Rtrue, Rmle));
fprintf(' MSE Rmle+: %g\n', synchromse(synchroproblem, Rtrue, Rmleplus));
fprintf('Noise parameters estimated by Rmle+:\n');
disp(params);

% Compute the CRB for this problem setup.
fprintf('Computing the Cram�r-Rao bound... ');
[C msecrb] = synchrocrb(synchroproblem);
fprintf('done.\n');
fprintf('Cram�r-Rao lower bound on the expected MSE: %g\n', msecrb);


% If you wish, you may also compute the CRB based on the noise parameters
% estimated by MLE+. For real data, this is the best you can do anyway.
fprintf('Computing the Cram�r-Rao bound with MLE+ parameters... ');
synchroproblemplus = changeproblemweights(synchroproblem, ...
                                   params.kappa1, params.kappa2, params.p);
[Cplus msecrbplus] = synchrocrb(synchroproblemplus);
fprintf('done.\n');
fprintf('Cram�r-Rao lower bound on the expected MSE, with MLE+ parameters: %g\n', msecrbplus);
