function isconn = isconnected(N, I, J)
% FUNCTION ISCONN = ISCONNECTED(N, I, J)
%
% Given a list of edges (I(k) <-> J(k)) in an undirected graph with N nodes
% numbered from 1 to N, this function returns true if the graph is
% connected, and false otherwise.
%
% This function does not check whether edges are specified only once or
% not. In particular, you should make sure that if the edge (i, j) is
% specified, the edge (j, i) does not appear in the edges list.
%
% TODO: This algorithm uses the Laplacian of the graph, but really, we
%       should just do a flood filling...
%
% Nicolas Boumal, UCLouvain, Sept. 15, 2011.

    assert(all(I~=J), 'There should be no edge from a node to itself.');
    
    M = length(I);
    assert(length(J) == M);
    
    % If there is only one node, the graph is connected by definition
    if N == 1
        isconn = true;
        return;
    end
    
    % If there are not enough edges to form a tree, the graph is certainly
    % disconnected
    if M < N-1
        isconn = false;
        return;
    end
    
    % Compute the graph Laplacian (sparse matrix)
%     L = laplacian(N, I, J);
    
    % The graph is connected if the multiplicity of the 0 eigenvalue of L
    % is one, or, equivalently, if the matrix L from which we deleted any
    % one row and column is full rank. For sparse matrices, this can be
    % checked by making sure that the smallest eigenvalue of such a matrix
    % is not 'too close to zero'.
%     cn = condest(L(2:end, 2:end));
%     isconn = cn < 1e10;
%     ev = eigs(L(2:end,2:end), 1, 0);
%     isconn = (ev > N*eps);

    % The graph is connected if it contains at least one tree (this may not
    % be the most efficient way, but it looks quite robust and is decently
    % fast)
    nbst = nbspanningtrees(N, I, J);
    isconn = (nbst > .5);
    
end
