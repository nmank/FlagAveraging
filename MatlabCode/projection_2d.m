function [y_hat] = projection_2d(y, y1, y2)

x = [y1, y2];

p_mat = eye(4)-x*x';

y_hat = p_mat * y;