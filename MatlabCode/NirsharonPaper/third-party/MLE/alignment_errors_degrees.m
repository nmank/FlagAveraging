function errs = alignment_errors_degrees(Rtrue, Rhat, Q)
% Given reference rotations, other rotations and an alignment matrix (best
% obtained from soregister(Rtrue, Rhat)), returns a vector with as many
% numbers as there are rotations, each one indicating in degrees the
% distance between the reference and the other rotation aligned by Q.
%
% Nicolas Boumal, UCLouvain, Oct. 9, 2013.

    RtR = multiprod(multitransp(Rtrue), multiprod(Rhat, Q));
    errs = zeros(size(RtR, 3), 1);
    for i = 1 : length(errs)
        errs(i) = norm(logm(RtR(:, :, i)), 'fro')*sqrt(2)*180/pi;
    end

end
