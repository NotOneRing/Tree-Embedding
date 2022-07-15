gcc -O3 -m64 -I/usr/include/eigen3 -I/opt/intel/oneapi/mkl/2021.3.0/include -I../include frpca.c Tree_d_NC.cpp matrix_vector_functions_intel_mkl_ext.c matrix_vector_functions_intel_mkl.c -Wl,--start-group /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_thread.a /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_core.a -Wl,--end-group -L/opt/intel/oneapi/compiler/2021.3.0/linux/compiler/lib/intel64_lin -liomp5 -lpthread -ldl -lm -fopenmp -w -o TREE_D_NC -lstdc++ 

gcc -O3 -m64 -I/usr/include/eigen3 -I/opt/intel/oneapi/mkl/2021.3.0/include -I../include frpca.c Tree_u_NC.cpp matrix_vector_functions_intel_mkl_ext.c matrix_vector_functions_intel_mkl.c -Wl,--start-group /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_thread.a /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_core.a -Wl,--end-group -L/opt/intel/oneapi/compiler/2021.3.0/linux/compiler/lib/intel64_lin -liomp5 -lpthread -ldl -lm -fopenmp -w -o TREE_U_NC -lstdc++ 

gcc -O3 -m64 -I/usr/include/eigen3 -I/opt/intel/oneapi/mkl/2021.3.0/include -I../include frpca.c Tree_d_LP.cpp matrix_vector_functions_intel_mkl_ext.c matrix_vector_functions_intel_mkl.c -Wl,--start-group /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_thread.a /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_core.a -Wl,--end-group -L/opt/intel/oneapi/compiler/2021.3.0/linux/compiler/lib/intel64_lin -liomp5 -lpthread -ldl -lm -fopenmp -w -o TREE_D_LP -lstdc++

gcc -O3 -m64 -I/usr/include/eigen3 -I/opt/intel/oneapi/mkl/2021.3.0/include -I../include frpca.c Tree_u_LP.cpp matrix_vector_functions_intel_mkl_ext.c matrix_vector_functions_intel_mkl.c -Wl,--start-group /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_lp64.a /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_intel_thread.a /opt/intel/oneapi/mkl/2021.3.0/lib/intel64/libmkl_core.a -Wl,--end-group -L/opt/intel/oneapi/compiler/2021.3.0/linux/compiler/lib/intel64_lin -liomp5 -lpthread -ldl -lm -fopenmp -w -o TREE_U_LP -lstdc++




