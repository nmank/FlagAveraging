% "test_error_calc"

n = 10;
d = 4;
l = 3;

A = make_data_MMG(d,l,n);
B = make_data_MMG(d,l,n);


[avg_sqr_err_V1, g ] = error_calc_MMG_V1(A, B);

[avg_sqr_err_V2, g_v2 ] = error_calc_MMG_V2(A, B);

[avg_sqr_err, shift_element] = error_calc_MMG(A, B);

[avg_sqr_err_V1, avg_sqr_err_V2, avg_sqr_err ]

