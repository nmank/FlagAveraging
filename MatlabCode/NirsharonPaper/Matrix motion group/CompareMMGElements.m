function [ norm_diff ] = CompareMMGElements(A, B)
% compare two elements of MMG
n = size(A,1);
norm_diff = 0;
for j=1:n
    norm_diff = norm_diff + norm(A{j,1}-B{j,1},'fro') + norm(A{j,2}-B{j,2},'fro') + norm(A{j,3}-B{j,3},'fro');
end

end

