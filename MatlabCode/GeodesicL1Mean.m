function R = GeodesicL1Mean(R_input, b_outlier_rejection, n_iterations, thr_convergence)
    
    % 1. Initialize
    
    n_samples = length(R_input);
    
    vectors_total = zeros(9,n_samples);
    for i = 1:n_samples
        vectors_total(:,i)= R_input{i}(:);
    end
    s = median(vectors_total,2);
    
    [U,~,V] = svd(reshape(s, [3 3]));
    R = U*V.';
    if (det(R) < 0)
        V(:,3) = -V(:,3);
        R = U*V.';
    end
    
    % 2. Optimize
    
    for j = 1:n_iterations

        vs = zeros(3,n_samples);
        v_norms = zeros(1,n_samples);
        for i = 1:n_samples
            v =  logarithm_map(R_input{i}*R');
            v_norm = norm(v);
            vs(:,i) = v;
            v_norms(i) = v_norm;
        end
        
        % Compute the inlier threshold (if we reject outliers).
        thr = inf;
        if (b_outlier_rejection)
            sorted_v_norms = sort(v_norms);
            v_norm_firstQ = sorted_v_norms(ceil(n_samples/4));
            if (n_samples <= 50)
                thr = max(v_norm_firstQ, 1);

            else
                thr = max(v_norm_firstQ, 0.5);
            end
        end

        step_num = 0;
        step_den = 0;

        for i = 1:n_samples
            v =  vs(:,i);
            v_norm = v_norms(i);
            if (v_norm > thr)
                continue;
            end
            step_num = step_num + v/v_norm;
            step_den = step_den + 1/v_norm;
        end

        delta = step_num/step_den;
        delta_angle = norm(delta);
        delta_axis = delta/delta_angle;
        
        R_delta = RotationFromUnitAxisAngle(delta_axis, delta_angle);
        R = R_delta*R;
        if (delta_angle < thr_convergence)
            break;
        end
    end
end