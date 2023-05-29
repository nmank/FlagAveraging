function [ estimations ] = sync_SE3_by_quatDiff(H, W )
% wrapping the diffusion quaternions synchronization

n = size(H,1)/4;
i = 1;
pairwise_motions = [];
for j=1:n
    for l=1:n
        if and(j~=l,W(j,l)~=0)
            pairwise_motions(i).i1 = double( j-1 );
            pairwise_motions(i).i2 = double( l-1 );
            current_m = H((4*j-3):(4*j),(4*l-3):(4*l));
            pairwise_motions(i).R = double( current_m(1:3,1:3));
            pairwise_motions(i).T = double( current_m(1:3,4)');
            i=i+1;
        end
    end
end

n_rangemaps = n; 
absolute_diffused = diffuse_dualquat(pairwise_motions, n_rangemaps);

estimations = zeros(4,4,n);
for j=1:n
    estimations(1:3,1:3,j) = absolute_diffused(j).R;
    estimations(1:3,4,j) = absolute_diffused(j).T';
    estimations(4,4,j) = 1;
end

end

