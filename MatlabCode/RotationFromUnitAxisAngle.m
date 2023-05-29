function R = RotationFromUnitAxisAngle(unit_axis, angle)
    
    if (angle==0)
        R = eye(3);
    else
        so3 = SkewSymmetricMatrix(unit_axis);
        R = eye(3)+so3*sin(angle)+so3^2*(1-cos(angle));
    end
end