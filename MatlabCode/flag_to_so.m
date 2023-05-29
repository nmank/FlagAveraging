function [ so_point ] = flag_to_so(flag_point)


new_vec = rand(4,1);

for i=1:3 
   new_vec_proj = flag_point(:,i)*flag_point(:,i)'*new_vec;
   new_vec = new_vec - new_vec_proj;
end

so_point = zeros(4,4);
so_point(:,1:3) = flag_point;
% so_point(:,4) = new_vec - new_vec_proj;
so_point(:,4) = new_vec/norm(new_vec);
% so_point(:,4) = so_point(:,4)/norm(so_point(:,4));

if det(so_point)< 0
    so_point(:,4) = -so_point(:,4);
end
