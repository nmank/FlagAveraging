function [R] = project_onto_so3(R)

[U, ~, V] = svd(R);
R = V*U';
if det(R)<0
   A = eye(3);
   A(3,3) = det(V*U');
   R = V*A*U';
end
%
% if (det(R) < 0)
%     V(:,3) = -V(:,3);
%     R = U*V.';
% end

end