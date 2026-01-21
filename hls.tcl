open_project mm_prj
set_top matmul_hls
add_files hls_version.cpp
open_solution "sol1"
set_part {xcu280-fsvh2892-2L-e}   ;# 例如 U280
create_clock -period 3.33        ;# 300MHz
csynth_design
exit