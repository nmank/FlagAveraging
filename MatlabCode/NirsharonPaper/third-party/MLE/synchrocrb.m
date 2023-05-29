function [C msecrb] = synchrocrb(problem)
% Returns the Cramér-Rao lower bound matrix for the synchronization problem.
%
% See the associate paper:
% http://imaiai.oxfordjournals.org/content/early/2013/09/23/imaiai.iat006.abstract
% N. Boumal, A. Singer, P.-A. Absil and V. D. Blondel,
% Cramér-Rao bounds for synchronization of rotations,
% in Information and Inference, a journal of the IMA, 2013.
%
% C is of size NxN and is such that C(i, i) is a lower bound on the
% expected squared distance between rotation i and the estimator of
% rotation i. This is valid for anchored synchronization. For anchor-free
% synchronization or 1-anchor synchronization,
% C(i, i) + C(j, j) - 2*C(i, j) is approximately a lower bound on the
% expected squared distance between the relative rotation Ri*Rj^T and the
% estimator of that relative rotation.
%
% This code for the CRB assumes the measurement graph is connected. If the
% graph is not connected (or not numerically connected in case of very
% small weights), the behaviour of this function is undefined. Most likely,
% there will be an error or a warning when calling the 'inv' function.
%
% For SO(2), there are no curvature terms in the CRB.
% For SO(3), the curvature terms are included in the CRB.
% For SO(n), n >= 4, the curvature terms are neglected in the CRB.
%
% msecrb is the CRB on the mean squared error (MSE), normalized by the
% number of rotations to estimate. If there are no anchors, the number of
% "free" rotations is N-1 and not N, because of the global rotation
% ambiguity.
%
% See also: synchromse synchrolaplacian
%
% Nicolas Boumal, UCLouvain, Jan. 16, 2013.

    n = problem.n;
    N = problem.N;
    A = problem.A;
    
    % Construct the synchronization Laplacian, where edges are weighted by
    % information quality.
    L = synchrolaplacian(problem);
    
    % If this is anchor-free syncrhonization
    if isempty(A)    
        % The CRB is essentially the pseudoinverse of L.
        % It can be computed generically as follows:
        % iLA = pinv(full(L));
        % This is rather slow. A faster way for anchor-free synchronization,
        % following a suggestion by Kunal Chaudhury, is to leverage the
        % fact that we know the null space of L: it is spanned by the vector
        % of all ones (this is since we assume the graph is connected).
        % Hence, by shifting the Laplacian along the null space direction,
        % we make it invertible; calling inv is much faster than calling
        % pinv. Then, simply remove the null space component.
        iLA = inv(full(L) + ones(N)/N) - ones(N)/N;
        % If you have a disconnected measurement graph but you are aware of
        % it and know how to interpret the CRB then (see the paper), then
        % it is better to call pinv: that is the correct formulation for
        % the CRB when the graph is disconnected.
        
    % If this is anchored synchronization
    else
        % Similarly here, we could simply build the masked Laplacian LA by
        % forcing rows and columns corresponding to anchors to zero:
        % LA = L;
        % LA(:, A) = 0;
        % LA(A, :) = 0;
        % iLA = pinv(full(LA));
        % This code also works for anchor-free synchronozation by the way,
        % but this is rather slow again. Instead, we know that this will
        % give the same result as if we invert the non-masked entries of LA:
        notA = setdiff(1:N, A);
        iLA = zeros(size(L));
        iLA(notA, notA) = inv(full(L(notA, notA)));
        
    end
    
    
    % For n = 2, there is no curvature. For n = 3, we have an explicit
    % expression. For n >= 4, we have no expression (but the curvature
    % terms in the CRB should be negligible).
    switch n
        case 2
            C = iLA;
        case 3
            diLA = diag(diag(iLA));
            C = 9 * ( iLA - .25*(diLA*iLA + iLA*diLA) );
        otherwise
            d = nchoosek(n, 2);
            C = (d^2) * iLA;
    end
    
    % CRB on the MSE
    msecrb = trace(C) / (N-max(1, numel(A)));

end
