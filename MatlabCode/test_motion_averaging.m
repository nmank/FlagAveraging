%number of points
n_pts = 10;

%parameter of contraction
lambda = 20;

the_data = cell(10,2);

for i=1:n_pts

    % mu is rotation
    % b is translation
    k=3;
    
    %parameter of contraction
    lambda = 20;
    
    A = rand(k);
    [mu, ~] = qr(A);
    % positive determinant
    mu(:,k) = det(mu)*mu(:,k);
    
    b = rand(k,1);

    the_data{i,1} = mu;
    the_data{i,2} = b;

end

flag_data = zeros(4,3,n_pts);
for i=1:n_pts
    so_point = se_to_so(the_data{i,1}, the_data{i,2}, lambda);
    flag_data(:,:,i) = so_to_flag(so_point);
end


the_flag_mean = chordal_flag_mean(flag_data);

so_mean = flag_to_so(the_flag_mean);
[mu_mean, b_mean] = so_to_se(so_mean, lambda);
