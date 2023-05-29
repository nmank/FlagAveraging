function L = laplacian(N, I, J, w)
% L = laplacian(N, I, J, w)
% 
% L is the un-normalized Laplacian of the N-nodes undirected graph
% described by the edges I(k)<->J(k). L is a sparse matrix.
%
% w is the weight vector such that edge (I(k), J(k)) has weight w(k). If w
% is not specified, all edges have weight 1. If w is a scalar, all edges
% have weight w.
%
% Nicolas Boumal, UCLouvain, Oct. 12, 2011.

    I = I(:);
    J = J(:);

    % Number of edges
    M = length(I);
    
    % Weight vector
    if ~exist('w', 'var') || isempty(w)
        w = ones(M, 1);
    end
    if numel(w) == 1
        w = w * ones(M, 1);
    end
    w = w(:);
    
    assert(length(w) == M);
    
    % Adjacency matrix
    A = sparse([I;J], [J;I], [w;w], N, N, 2*M);
    
    % Degree matrix
    D = spdiags(A*ones(N, 1), 0, N, N);
    
    % Laplacian
    L = D-A;
    
end
