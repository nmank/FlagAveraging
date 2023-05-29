function [nbst lognbst] = nbspanningtrees(N, I, J)
% [nbst lognbst] = nbspanningtrees(N, I, J)
% 
% nbst is the number of spanning trees in the undirected simple graph with
% N nodes and edges defined by the vectors I and J. Based on Kirchhoff's
% matrix tree theorem. This function returns 0 (or a number very close to
% zero) if the graph is disconnected. The second output, lognbst, is the
% logarithm in base 10 of nbst, which should help when nbst overflows.
%
% Nicolas Boumal, UCLouvain, Oct. 12, 2011.

    % Laplacian
    L = laplacian(N, I, J);
    
    % The number of spanning trees is equal to the absolute value of any
    % cofactor of the Laplacian matrix.
    % We exploit the fact that the submatrix of the Laplacian is symmetric
    % positive definite to compute a sparse Cholesky factorization, which
    % automatically reveals the determinant (along with all the eigenvalues
    % actually). This has the added benefit of letting us compute the
    % logarithm of the determinant without overflow.

    [R, p, ~] = chol(L(2:end, 2:end), 'vector');
    if p == 0
        nbst = round(full(prod(diag(R).^2)));
        lognbst = full(sum(2*log10(diag(R))));
    else
        nbst = 0;
        lognbst = -inf;
    end
    
end