# BITWIDTH
# make run_strategy outdir=output/bitwidth/original        n_c=1 c_begin=0 c_end=1 hook=True strategy=original                                   analysis=bitwidth
# make run_strategy outdir=output/bitwidth/spatial         n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial           ref_first=True           analysis=bitwidth
# make run_strategy outdir=output/bitwidth/spatial         n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial           ref_first=False          analysis=bitwidth
# make run_strategy outdir=output/bitwidth/temporal        n_c=1 c_begin=0 c_end=1 hook=True strategy=temporal                                   analysis=bitwidth
# make run_strategy outdir=output/bitwidth/cfg             n_c=1 c_begin=0 c_end=1 hook=True strategy=cfg                                        analysis=bitwidth
# make run_strategy outdir=output/bitwidth/cfg_opt         n_c=1 c_begin=0 c_end=1 hook=True strategy=cfg_opt                           bit_th=4 analysis=bitwidth
# make run_strategy outdir=output/bitwidth/spatial_cfg     n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial_cfg       ref_first=True           analysis=bitwidth
# make run_strategy outdir=output/bitwidth/spatial_cfg     n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial_cfg       ref_first=False          analysis=bitwidth
# make run_strategy outdir=output/bitwidth/spatial_cfg_opt n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial_cfg_opt   ref_first=True  bit_th=4 analysis=bitwidth
# make run_strategy outdir=output/bitwidth/spatial_cfg_opt n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial_cfg_opt   ref_first=False bit_th=4 analysis=bitwidth
# make run_strategy outdir=output/bitwidth/Raw_SD          n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_SD            ref_first=False          analysis=bitwidth
# make run_strategy outdir=output/bitwidth/Raw_TD          n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_TD                                     analysis=bitwidth
    # make run_strategy strategy=cfg_large bit_th=3 analysis=bitwidth
    # make run_strategy strategy=cfg_large bit_th=4 analysis=bitwidth
    # make run_strategy strategy=cfg_large bit_th=5 analysis=bitwidth

# SIMILARITY
# make run_strategy hook=True strategy=original ref_first=True  analysis=similarity similarity_type=spatial
# make run_strategy hook=True strategy=original ref_first=False analysis=similarity similarity_type=spatial
# make run_strategy hook=True strategy=original                 analysis=similarity similarity_type=temporal
# make run_strategy hook=True strategy=original                 analysis=similarity similarity_type=conditional

# SHAPE
# make run_strategy n_c=1 c_begin=0 c_end=0 hook=True analysis=shape
# ==================================================


# BITSLICE (with bitslice_analysis.csv, 20s/img)
#! check outdir="Final"
#! check strategy save analyze_tensor input[0]
#* BADA
# make run_strategy outdir=output/bitslice/bada_int9_bs3/Raw  n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/bada_int9_bs3/SD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/bada_int9_bs3/TD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/bada_int9_bs3/CUD  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False

# make run_strategy outdir=output/bitslice2/bada_int9_bs3/Raw  n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/bada_int9_bs3/SD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/bada_int9_bs3/TD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/bada_int9_bs3/CUD  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False

# make run_strategy outdir=output/bitslice/bada_int9_bs3/original n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/bada_int9_bs3/Raw_SD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/bada_int9_bs3/Raw_TD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/bada_int9_bs3/Raw_CUD  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/bada_int9_bs3/SD_SD    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/bada_int9_bs3/SD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/bada_int9_bs3/SD_CUD   n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/bada_int9_bs3/TD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/bada_int9_bs3/TD_GD    n_c=1 c_begin=0 c_end=0 hook=True strategy=TD_GD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False

# make run_strategy outdir=output/bitslice2/bada_int9_bs3/original n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/bada_int9_bs3/Raw_SD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/bada_int9_bs3/Raw_TD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/bada_int9_bs3/Raw_CUD  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/bada_int9_bs3/SD_SD    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/bada_int9_bs3/SD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/bada_int9_bs3/SD_CUD   n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/bada_int9_bs3/TD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/bada_int9_bs3/TD_GD    n_c=1 c_begin=0 c_end=0 hook=True strategy=TD_GD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False

#* Naive
# make run_strategy outdir=output/bitslice/naive_int9_bs3/Raw  n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/naive_int9_bs3/SD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/naive_int9_bs3/TD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/naive_int9_bs3/CUD  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False

# make run_strategy outdir=output/bitslice2/naive_int9_bs3/Raw  n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/naive_int9_bs3/SD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/naive_int9_bs3/TD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/naive_int9_bs3/CUD  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False

# make run_strategy outdir=output/bitslice/naive_int9_bs3/original n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/naive_int9_bs3/Raw_SD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/naive_int9_bs3/Raw_TD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/naive_int9_bs3/Raw_CUD  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/naive_int9_bs3/SD_SD    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/naive_int9_bs3/SD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/naive_int9_bs3/SD_CUD   n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/naive_int9_bs3/TD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/naive_int9_bs3/TD_GD    n_c=1 c_begin=0 c_end=0 hook=True strategy=TD_GD       clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False

# make run_strategy outdir=output/bitslice2/naive_int9_bs3/original n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/naive_int9_bs3/Raw_SD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/naive_int9_bs3/Raw_TD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/naive_int9_bs3/Raw_CUD  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/naive_int9_bs3/SD_SD    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/naive_int9_bs3/SD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/naive_int9_bs3/SD_CUD   n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/naive_int9_bs3/TD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/naive_int9_bs3/TD_GD    n_c=1 c_begin=0 c_end=0 hook=True strategy=TD_GD       clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False

#* Sibia
# make run_strategy outdir=output/bitslice/sibia_int9_bs3/Raw  n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/sibia_int9_bs3/SD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/sibia_int9_bs3/TD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/sibia_int9_bs3/CUD  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False

# make run_strategy outdir=output/bitslice2/sibia_int9_bs3/Raw  n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/sibia_int9_bs3/SD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/sibia_int9_bs3/TD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/sibia_int9_bs3/CUD  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False

# make run_strategy outdir=output/bitslice/sibia_int9_bs3/original n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/sibia_int9_bs3/Raw_SD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/sibia_int9_bs3/Raw_TD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/sibia_int9_bs3/Raw_CUD  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/sibia_int9_bs3/SD_SD    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/sibia_int9_bs3/SD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/sibia_int9_bs3/SD_CUD   n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/sibia_int9_bs3/TD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make run_strategy outdir=output/bitslice/sibia_int9_bs3/TD_GD    n_c=1 c_begin=0 c_end=0 hook=True strategy=TD_GD       clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False

# make run_strategy outdir=output/bitslice2/sibia_int9_bs3/original n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/sibia_int9_bs3/Raw_SD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/sibia_int9_bs3/Raw_TD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/sibia_int9_bs3/Raw_CUD  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/sibia_int9_bs3/SD_SD    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/sibia_int9_bs3/SD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/sibia_int9_bs3/SD_CUD   n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/sibia_int9_bs3/TD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make run_strategy outdir=output/bitslice2/sibia_int9_bs3/TD_GD    n_c=1 c_begin=0 c_end=0 hook=True strategy=TD_GD       clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False

# ==================================================


# BITSLICE SUMMARY

#* BADA
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/Raw/Final  outdir=output/summary/bitslice/bada_int9_bs3/Raw  bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/SD/Final   outdir=output/summary/bitslice/bada_int9_bs3/SD   bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/TD/Final   outdir=output/summary/bitslice/bada_int9_bs3/TD   bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/CUD/Final  outdir=output/summary/bitslice/bada_int9_bs3/CUD  bitslice_overflow=True

# make summarize_bitslice indir=output/bitslice2/bada_int9_bs3/Raw/Final  outdir=output/summary/bitslice2/bada_int9_bs3/Raw  bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice2/bada_int9_bs3/SD/Final   outdir=output/summary/bitslice2/bada_int9_bs3/SD   bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice2/bada_int9_bs3/TD/Final   outdir=output/summary/bitslice2/bada_int9_bs3/TD   bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice2/bada_int9_bs3/CUD/Final  outdir=output/summary/bitslice2/bada_int9_bs3/CUD  bitslice_overflow=True

# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/original/Final outdir=output/summary/bitslice/bada_int9_bs3/original bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/Raw_SD/Final   outdir=output/summary/bitslice/bada_int9_bs3/Raw_SD   bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/Raw_TD/Final   outdir=output/summary/bitslice/bada_int9_bs3/Raw_TD   bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/Raw_CUD/Final  outdir=output/summary/bitslice/bada_int9_bs3/Raw_CUD  bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/SD_SD/Final    outdir=output/summary/bitslice/bada_int9_bs3/SD_SD    bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/SD_TD/Final    outdir=output/summary/bitslice/bada_int9_bs3/SD_TD    bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/SD_CUD/Final   outdir=output/summary/bitslice/bada_int9_bs3/SD_CUD   bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/TD_TD/Final    outdir=output/summary/bitslice/bada_int9_bs3/TD_TD    bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/TD_GD/Final    outdir=output/summary/bitslice/bada_int9_bs3/TD_GD    bitslice_overflow=True

# make summarize_bitslice indir=output/bitslice2/bada_int9_bs3/original/Final outdir=output/summary/bitslice2/bada_int9_bs3/original bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice2/bada_int9_bs3/Raw_SD/Final   outdir=output/summary/bitslice2/bada_int9_bs3/Raw_SD   bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice2/bada_int9_bs3/Raw_TD/Final   outdir=output/summary/bitslice2/bada_int9_bs3/Raw_TD   bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice2/bada_int9_bs3/Raw_CUD/Final  outdir=output/summary/bitslice2/bada_int9_bs3/Raw_CUD  bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice2/bada_int9_bs3/SD_SD/Final    outdir=output/summary/bitslice2/bada_int9_bs3/SD_SD    bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice2/bada_int9_bs3/SD_TD/Final    outdir=output/summary/bitslice2/bada_int9_bs3/SD_TD    bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice2/bada_int9_bs3/SD_CUD/Final   outdir=output/summary/bitslice2/bada_int9_bs3/SD_CUD   bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice2/bada_int9_bs3/TD_TD/Final    outdir=output/summary/bitslice2/bada_int9_bs3/TD_TD    bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice2/bada_int9_bs3/TD_GD/Final    outdir=output/summary/bitslice2/bada_int9_bs3/TD_GD    bitslice_overflow=True

#* Naive
# make summarize_bitslice indir=output/bitslice/naive_int9_bs3/Raw/Final  outdir=output/summary/bitslice/naive_int9_bs3/Raw  bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/naive_int9_bs3/SD/Final   outdir=output/summary/bitslice/naive_int9_bs3/SD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/naive_int9_bs3/TD/Final   outdir=output/summary/bitslice/naive_int9_bs3/TD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/naive_int9_bs3/CUD/Final  outdir=output/summary/bitslice/naive_int9_bs3/CUD  bitslice_overflow=False

# make summarize_bitslice indir=output/bitslice2/naive_int9_bs3/Raw/Final  outdir=output/summary/bitslice2/naive_int9_bs3/Raw  bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/naive_int9_bs3/SD/Final   outdir=output/summary/bitslice2/naive_int9_bs3/SD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/naive_int9_bs3/TD/Final   outdir=output/summary/bitslice2/naive_int9_bs3/TD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/naive_int9_bs3/CUD/Final  outdir=output/summary/bitslice2/naive_int9_bs3/CUD  bitslice_overflow=False

# make summarize_bitslice indir=output/bitslice/naive_int9_bs3/original/Final outdir=output/summary/bitslice/naive_int9_bs3/original bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/naive_int9_bs3/Raw_SD/Final   outdir=output/summary/bitslice/naive_int9_bs3/Raw_SD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/naive_int9_bs3/Raw_TD/Final   outdir=output/summary/bitslice/naive_int9_bs3/Raw_TD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/naive_int9_bs3/Raw_CUD/Final  outdir=output/summary/bitslice/naive_int9_bs3/Raw_CUD  bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/naive_int9_bs3/SD_SD/Final    outdir=output/summary/bitslice/naive_int9_bs3/SD_SD    bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/naive_int9_bs3/SD_TD/Final    outdir=output/summary/bitslice/naive_int9_bs3/SD_TD    bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/naive_int9_bs3/SD_CUD/Final   outdir=output/summary/bitslice/naive_int9_bs3/SD_CUD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/naive_int9_bs3/TD_TD/Final    outdir=output/summary/bitslice/naive_int9_bs3/TD_TD    bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/naive_int9_bs3/TD_GD/Final    outdir=output/summary/bitslice/naive_int9_bs3/TD_GD    bitslice_overflow=False

# make summarize_bitslice indir=output/bitslice2/naive_int9_bs3/original/Final outdir=output/summary/bitslice2/naive_int9_bs3/original bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/naive_int9_bs3/Raw_SD/Final   outdir=output/summary/bitslice2/naive_int9_bs3/Raw_SD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/naive_int9_bs3/Raw_TD/Final   outdir=output/summary/bitslice2/naive_int9_bs3/Raw_TD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/naive_int9_bs3/Raw_CUD/Final  outdir=output/summary/bitslice2/naive_int9_bs3/Raw_CUD  bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/naive_int9_bs3/SD_SD/Final    outdir=output/summary/bitslice2/naive_int9_bs3/SD_SD    bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/naive_int9_bs3/SD_TD/Final    outdir=output/summary/bitslice2/naive_int9_bs3/SD_TD    bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/naive_int9_bs3/SD_CUD/Final   outdir=output/summary/bitslice2/naive_int9_bs3/SD_CUD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/naive_int9_bs3/TD_TD/Final    outdir=output/summary/bitslice2/naive_int9_bs3/TD_TD    bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/naive_int9_bs3/TD_GD/Final    outdir=output/summary/bitslice2/naive_int9_bs3/TD_GD    bitslice_overflow=False

#* Sibia
# make summarize_bitslice indir=output/bitslice/sibia_int9_bs3/Raw/Final  outdir=output/summary/bitslice/sibia_int9_bs3/Raw  bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/sibia_int9_bs3/SD/Final   outdir=output/summary/bitslice/sibia_int9_bs3/SD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/sibia_int9_bs3/TD/Final   outdir=output/summary/bitslice/sibia_int9_bs3/TD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/sibia_int9_bs3/CUD/Final  outdir=output/summary/bitslice/sibia_int9_bs3/CUD  bitslice_overflow=False

# make summarize_bitslice indir=output/bitslice2/sibia_int9_bs3/Raw/Final  outdir=output/summary/bitslice2/sibia_int9_bs3/Raw  bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/sibia_int9_bs3/SD/Final   outdir=output/summary/bitslice2/sibia_int9_bs3/SD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/sibia_int9_bs3/TD/Final   outdir=output/summary/bitslice2/sibia_int9_bs3/TD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/sibia_int9_bs3/CUD/Final  outdir=output/summary/bitslice2/sibia_int9_bs3/CUD  bitslice_overflow=False

# make summarize_bitslice indir=output/bitslice/sibia_int9_bs3/original/Final outdir=output/summary/bitslice/sibia_int9_bs3/original bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/sibia_int9_bs3/Raw_SD/Final   outdir=output/summary/bitslice/sibia_int9_bs3/Raw_SD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/sibia_int9_bs3/Raw_TD/Final   outdir=output/summary/bitslice/sibia_int9_bs3/Raw_TD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/sibia_int9_bs3/Raw_CUD/Final  outdir=output/summary/bitslice/sibia_int9_bs3/Raw_CUD  bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/sibia_int9_bs3/SD_SD/Final    outdir=output/summary/bitslice/sibia_int9_bs3/SD_SD    bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/sibia_int9_bs3/SD_TD/Final    outdir=output/summary/bitslice/sibia_int9_bs3/SD_TD    bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/sibia_int9_bs3/SD_CUD/Final   outdir=output/summary/bitslice/sibia_int9_bs3/SD_CUD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/sibia_int9_bs3/TD_TD/Final    outdir=output/summary/bitslice/sibia_int9_bs3/TD_TD    bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice/sibia_int9_bs3/TD_GD/Final    outdir=output/summary/bitslice/sibia_int9_bs3/TD_GD    bitslice_overflow=False

# make summarize_bitslice indir=output/bitslice2/sibia_int9_bs3/original/Final outdir=output/summary/bitslice2/sibia_int9_bs3/original bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/sibia_int9_bs3/Raw_SD/Final   outdir=output/summary/bitslice2/sibia_int9_bs3/Raw_SD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/sibia_int9_bs3/Raw_TD/Final   outdir=output/summary/bitslice2/sibia_int9_bs3/Raw_TD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/sibia_int9_bs3/Raw_CUD/Final  outdir=output/summary/bitslice2/sibia_int9_bs3/Raw_CUD  bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/sibia_int9_bs3/SD_SD/Final    outdir=output/summary/bitslice2/sibia_int9_bs3/SD_SD    bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/sibia_int9_bs3/SD_TD/Final    outdir=output/summary/bitslice2/sibia_int9_bs3/SD_TD    bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/sibia_int9_bs3/SD_CUD/Final   outdir=output/summary/bitslice2/sibia_int9_bs3/SD_CUD   bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/sibia_int9_bs3/TD_TD/Final    outdir=output/summary/bitslice2/sibia_int9_bs3/TD_TD    bitslice_overflow=False
# make summarize_bitslice indir=output/bitslice2/sibia_int9_bs3/TD_GD/Final    outdir=output/summary/bitslice2/sibia_int9_bs3/TD_GD    bitslice_overflow=False

# ==================================================


# BITWIDTH (with int9_full_distribution.csv, 60s/img)
# make run_strategy outdir=output/bitwidth/Raw      n_c=1 c_begin=0 c_end=0 hook=True strategy=original    analysis=bitwidth path_reverse=False
# make run_strategy outdir=output/bitwidth/SD       n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      analysis=bitwidth path_reverse=False
# make run_strategy outdir=output/bitwidth/TD       n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      analysis=bitwidth path_reverse=False
# make run_strategy outdir=output/bitwidth/CUD      n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         analysis=bitwidth path_reverse=False

# make run_strategy outdir=output/bitwidth2/Raw      n_c=1 c_begin=0 c_end=0 hook=True strategy=original    analysis=bitwidth path_reverse=True
# make run_strategy outdir=output/bitwidth2/SD       n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      analysis=bitwidth path_reverse=True
# make run_strategy outdir=output/bitwidth2/TD       n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      analysis=bitwidth path_reverse=True
# make run_strategy outdir=output/bitwidth2/CUD      n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         analysis=bitwidth path_reverse=True

# make run_strategy outdir=output/bitwidth/original n_c=1 c_begin=0 c_end=0 hook=True strategy=original    analysis=bitwidth
# make run_strategy outdir=output/bitwidth/Raw_SD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      analysis=bitwidth
# make run_strategy outdir=output/bitwidth/Raw_TD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      analysis=bitwidth
# make run_strategy outdir=output/bitwidth/Raw_CUD  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         analysis=bitwidth
# make run_strategy outdir=output/bitwidth/SD_SD    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     analysis=bitwidth
# make run_strategy outdir=output/bitwidth/SD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       analysis=bitwidth
# make run_strategy outdir=output/bitwidth/SD_CUD   n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg analysis=bitwidth
# make run_strategy outdir=output/bitwidth/TD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    analysis=bitwidth
# ==================================================


# BITWIDTH SUMMARY
# make summarize_bitwidth indir=output/bitwidth/Raw/Final      outdir=output/summary/bitwidth/Raw
# make summarize_bitwidth indir=output/bitwidth/SD/Final       outdir=output/summary/bitwidth/SD 
# make summarize_bitwidth indir=output/bitwidth/TD/Final       outdir=output/summary/bitwidth/TD 
# make summarize_bitwidth indir=output/bitwidth/CUD/Final      outdir=output/summary/bitwidth/CUD

# make summarize_bitwidth indir=output/bitwidth2/Raw/Final      outdir=output/summary/bitwidth2/Raw
# make summarize_bitwidth indir=output/bitwidth2/SD/Final       outdir=output/summary/bitwidth2/SD 
# make summarize_bitwidth indir=output/bitwidth2/TD/Final       outdir=output/summary/bitwidth2/TD 
# make summarize_bitwidth indir=output/bitwidth2/CUD/Final      outdir=output/summary/bitwidth2/CUD

# make summarize_bitwidth indir=output/bitwidth/original/Final outdir=output/summary/bitwidth/original
# make summarize_bitwidth indir=output/bitwidth/Raw_SD/Final   outdir=output/summary/bitwidth/Raw_SD
# make summarize_bitwidth indir=output/bitwidth/Raw_TD/Final   outdir=output/summary/bitwidth/Raw_TD
# make summarize_bitwidth indir=output/bitwidth/Raw_CUD/Final  outdir=output/summary/bitwidth/Raw_CUD
# make summarize_bitwidth indir=output/bitwidth/SD_SD/Final    outdir=output/summary/bitwidth/SD_SD
# make summarize_bitwidth indir=output/bitwidth/SD_TD/Final    outdir=output/summary/bitwidth/SD_TD
# make summarize_bitwidth indir=output/bitwidth/SD_CUD/Final   outdir=output/summary/bitwidth/SD_CUD
# make summarize_bitwidth indir=output/bitwidth/TD_TD/Final    outdir=output/summary/bitwidth/TD_TD
# ==================================================


# TILE SAVING
#! check save="True"
# make run_strategy outdir=output/patterns/original n_c=1 c_begin=0 c_end=0 hook=True strategy=original    save_tile=True path_reverse=False
# make run_strategy outdir=output/patterns/Raw_SD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      save_tile=True path_reverse=False
# make run_strategy outdir=output/patterns/Raw_TD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      save_tile=True path_reverse=False
# make run_strategy outdir=output/patterns/Raw_CUD  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         save_tile=True path_reverse=False
# make run_strategy outdir=output/patterns/SD_SD    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     save_tile=True path_reverse=False
# make run_strategy outdir=output/patterns/SD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       save_tile=True path_reverse=False
# make run_strategy outdir=output/patterns/SD_CUD   n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg save_tile=True path_reverse=False
# make run_strategy outdir=output/patterns/TD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    save_tile=True path_reverse=False

# make run_strategy outdir=output/patterns/original n_c=1 c_begin=0 c_end=0 hook=True strategy=original    save_tile=True path_reverse=True
# make run_strategy outdir=output/patterns/Raw_SD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      save_tile=True path_reverse=True
# make run_strategy outdir=output/patterns/Raw_TD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      save_tile=True path_reverse=True
# make run_strategy outdir=output/patterns/Raw_CUD  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         save_tile=True path_reverse=True
# make run_strategy outdir=output/patterns/SD_SD    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     save_tile=True path_reverse=True
# make run_strategy outdir=output/patterns/SD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       save_tile=True path_reverse=True
# make run_strategy outdir=output/patterns/SD_CUD   n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg save_tile=True path_reverse=True
# make run_strategy outdir=output/patterns/TD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    save_tile=True path_reverse=True


# ==================================================


# PERFORMANCE (with performance_analysis.csv, 70s/img)
#! check scheme in bitslice_analyzer.py, and no normal bitslice anlayze (in create_strategy)




#* PATH_REVERSE=FALSE
# make run_strategy outdir=output/performance/bada_int9_bs3/original/scheme1  n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/Raw_SD/scheme1    n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/Raw_TD/scheme1    n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/Raw_CUD/scheme1   n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/SD_SD/scheme1     n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/SD_TD/scheme1     n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/SD_CUD/scheme1    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/TD_TD/scheme1     n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/TD_GD/scheme1     n_c=1 c_begin=0 c_end=0 hook=True strategy=TD_GD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True

# make run_strategy outdir=output/performance/bada_int9_bs3/original/scheme2 n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/Raw_SD/scheme2   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/Raw_TD/scheme2   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/Raw_CUD/scheme2  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/SD_SD/scheme2    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/SD_TD/scheme2    n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/SD_CUD/scheme2   n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/TD_TD/scheme2    n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/TD_GD/scheme2    n_c=1 c_begin=0 c_end=0 hook=True strategy=TD_GD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True

#* mem
# make run_strategy outdir=output/performance/bada_int9_bs3/original/mem/scheme1  n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/Raw_SD/mem/scheme1    n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/Raw_TD/mem/scheme1    n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/Raw_CUD/mem/scheme1   n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/SD_SD/mem/scheme1     n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/SD_TD/mem/scheme1     n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/SD_CUD/mem/scheme1    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/TD_TD/mem/scheme1     n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True

# make run_strategy outdir=output/performance/bada_int9_bs3/original/mem/scheme2 n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/Raw_SD/mem/scheme2   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/Raw_TD/mem/scheme2   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/Raw_CUD/mem/scheme2  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/SD_SD/mem/scheme2    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/SD_TD/mem/scheme2    n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/SD_CUD/mem/scheme2   n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make run_strategy outdir=output/performance/bada_int9_bs3/TD_TD/mem/scheme2    n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True

#* PATH_REVERSE=TRUE
# make run_strategy outdir=output/performance2/bada_int9_bs3/original/scheme1 n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/Raw_SD/scheme1   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/Raw_TD/scheme1   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/Raw_CUD/scheme1  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/SD_SD/scheme1    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/SD_TD/scheme1    n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/SD_CUD/scheme1   n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/TD_TD/scheme1    n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/TD_GD/scheme1    n_c=1 c_begin=0 c_end=0 hook=True strategy=TD_GD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True

# make run_strategy outdir=output/performance2/bada_int9_bs3/original/scheme2 n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/Raw_SD/scheme2   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/Raw_TD/scheme2   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/Raw_CUD/scheme2  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/SD_SD/scheme2    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/SD_TD/scheme2    n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/SD_CUD/scheme2   n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/TD_TD/scheme2    n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/TD_GD/scheme2    n_c=1 c_begin=0 c_end=0 hook=True strategy=TD_GD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True

#* mem
# make run_strategy outdir=output/performance2/bada_int9_bs3/original/mem/scheme1 n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/Raw_SD/mem/scheme1   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/Raw_TD/mem/scheme1   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/Raw_CUD/mem/scheme1  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/SD_SD/mem/scheme1    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/SD_TD/mem/scheme1    n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/SD_CUD/mem/scheme1   n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/TD_TD/mem/scheme1    n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True

# make run_strategy outdir=output/performance2/bada_int9_bs3/original/mem/scheme2 n_c=1 c_begin=0 c_end=0 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/Raw_SD/mem/scheme2   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/Raw_TD/mem/scheme2   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/Raw_CUD/mem/scheme2  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/SD_SD/mem/scheme2    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/SD_TD/mem/scheme2    n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/SD_CUD/mem/scheme2   n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make run_strategy outdir=output/performance2/bada_int9_bs3/TD_TD/mem/scheme2    n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# ==================================================


# PERFORMANCE SUMMARY
#* PATH_REVERSE=FALSE
# make summarize_performance indir=output/performance/bada_int9_bs3/original/scheme1/Final outdir=output/summary/performance/bada_int9_bs3/original/scheme1
# make summarize_performance indir=output/performance/bada_int9_bs3/Raw_SD/scheme1/Final   outdir=output/summary/performance/bada_int9_bs3/Raw_SD/scheme1  
# make summarize_performance indir=output/performance/bada_int9_bs3/Raw_TD/scheme1/Final   outdir=output/summary/performance/bada_int9_bs3/Raw_TD/scheme1  
# make summarize_performance indir=output/performance/bada_int9_bs3/Raw_CUD/scheme1/Final  outdir=output/summary/performance/bada_int9_bs3/Raw_CUD/scheme1 
# make summarize_performance indir=output/performance/bada_int9_bs3/SD_SD/scheme1/Final    outdir=output/summary/performance/bada_int9_bs3/SD_SD/scheme1   
# make summarize_performance indir=output/performance/bada_int9_bs3/SD_TD/scheme1/Final    outdir=output/summary/performance/bada_int9_bs3/SD_TD/scheme1   
# make summarize_performance indir=output/performance/bada_int9_bs3/SD_CUD/scheme1/Final   outdir=output/summary/performance/bada_int9_bs3/SD_CUD/scheme1  
# make summarize_performance indir=output/performance/bada_int9_bs3/TD_TD/scheme1/Final    outdir=output/summary/performance/bada_int9_bs3/TD_TD/scheme1   
# make summarize_performance indir=output/performance/bada_int9_bs3/TD_GD/scheme1/Final    outdir=output/summary/performance/bada_int9_bs3/TD_GD/scheme1   

# make summarize_performance indir=output/performance/bada_int9_bs3/original/scheme2/Final outdir=output/summary/performance/bada_int9_bs3/original/scheme2
# make summarize_performance indir=output/performance/bada_int9_bs3/Raw_SD/scheme2/Final   outdir=output/summary/performance/bada_int9_bs3/Raw_SD/scheme2  
# make summarize_performance indir=output/performance/bada_int9_bs3/Raw_TD/scheme2/Final   outdir=output/summary/performance/bada_int9_bs3/Raw_TD/scheme2  
# make summarize_performance indir=output/performance/bada_int9_bs3/Raw_CUD/scheme2/Final  outdir=output/summary/performance/bada_int9_bs3/Raw_CUD/scheme2 
# make summarize_performance indir=output/performance/bada_int9_bs3/SD_SD/scheme2/Final    outdir=output/summary/performance/bada_int9_bs3/SD_SD/scheme2   
# make summarize_performance indir=output/performance/bada_int9_bs3/SD_TD/scheme2/Final    outdir=output/summary/performance/bada_int9_bs3/SD_TD/scheme2   
# make summarize_performance indir=output/performance/bada_int9_bs3/SD_CUD/scheme2/Final   outdir=output/summary/performance/bada_int9_bs3/SD_CUD/scheme2  
# make summarize_performance indir=output/performance/bada_int9_bs3/TD_TD/scheme2/Final    outdir=output/summary/performance/bada_int9_bs3/TD_TD/scheme2   
# make summarize_performance indir=output/performance/bada_int9_bs3/TD_GD/scheme2/Final    outdir=output/summary/performance/bada_int9_bs3/TD_GD/scheme2   

#* mem
# make summarize_performance indir=output/performance/bada_int9_bs3/original/mem/scheme1/Final outdir=output/summary/performance/bada_int9_bs3/original/mem/scheme1
# make summarize_performance indir=output/performance/bada_int9_bs3/Raw_SD/mem/scheme1/Final   outdir=output/summary/performance/bada_int9_bs3/Raw_SD/mem/scheme1  
# make summarize_performance indir=output/performance/bada_int9_bs3/Raw_TD/mem/scheme1/Final   outdir=output/summary/performance/bada_int9_bs3/Raw_TD/mem/scheme1  
# make summarize_performance indir=output/performance/bada_int9_bs3/Raw_CUD/mem/scheme1/Final  outdir=output/summary/performance/bada_int9_bs3/Raw_CUD/mem/scheme1 
# make summarize_performance indir=output/performance/bada_int9_bs3/SD_SD/mem/scheme1/Final    outdir=output/summary/performance/bada_int9_bs3/SD_SD/mem/scheme1   
# make summarize_performance indir=output/performance/bada_int9_bs3/SD_TD/mem/scheme1/Final    outdir=output/summary/performance/bada_int9_bs3/SD_TD/mem/scheme1   
# make summarize_performance indir=output/performance/bada_int9_bs3/SD_CUD/mem/scheme1/Final   outdir=output/summary/performance/bada_int9_bs3/SD_CUD/mem/scheme1  
# make summarize_performance indir=output/performance/bada_int9_bs3/TD_TD/mem/scheme1/Final    outdir=output/summary/performance/bada_int9_bs3/TD_TD/mem/scheme1   

# make summarize_performance indir=output/performance/bada_int9_bs3/original/mem/scheme2/Final outdir=output/summary/performance/bada_int9_bs3/original/mem/scheme2
# make summarize_performance indir=output/performance/bada_int9_bs3/Raw_SD/mem/scheme2/Final   outdir=output/summary/performance/bada_int9_bs3/Raw_SD/mem/scheme2  
# make summarize_performance indir=output/performance/bada_int9_bs3/Raw_TD/mem/scheme2/Final   outdir=output/summary/performance/bada_int9_bs3/Raw_TD/mem/scheme2  
# make summarize_performance indir=output/performance/bada_int9_bs3/Raw_CUD/mem/scheme2/Final  outdir=output/summary/performance/bada_int9_bs3/Raw_CUD/mem/scheme2 
# make summarize_performance indir=output/performance/bada_int9_bs3/SD_SD/mem/scheme2/Final    outdir=output/summary/performance/bada_int9_bs3/SD_SD/mem/scheme2   
# make summarize_performance indir=output/performance/bada_int9_bs3/SD_TD/mem/scheme2/Final    outdir=output/summary/performance/bada_int9_bs3/SD_TD/mem/scheme2   
# make summarize_performance indir=output/performance/bada_int9_bs3/SD_CUD/mem/scheme2/Final   outdir=output/summary/performance/bada_int9_bs3/SD_CUD/mem/scheme2  
# make summarize_performance indir=output/performance/bada_int9_bs3/TD_TD/mem/scheme2/Final    outdir=output/summary/performance/bada_int9_bs3/TD_TD/mem/scheme2   

#* PATH_REVERSE=TRUE
# make summarize_performance indir=output/performance2/bada_int9_bs3/original/scheme1/Final outdir=output/summary/performance2/bada_int9_bs3/original/scheme1
# make summarize_performance indir=output/performance2/bada_int9_bs3/Raw_SD/scheme1/Final   outdir=output/summary/performance2/bada_int9_bs3/Raw_SD/scheme1  
# make summarize_performance indir=output/performance2/bada_int9_bs3/Raw_TD/scheme1/Final   outdir=output/summary/performance2/bada_int9_bs3/Raw_TD/scheme1  
# make summarize_performance indir=output/performance2/bada_int9_bs3/Raw_CUD/scheme1/Final  outdir=output/summary/performance2/bada_int9_bs3/Raw_CUD/scheme1 
# make summarize_performance indir=output/performance2/bada_int9_bs3/SD_SD/scheme1/Final    outdir=output/summary/performance2/bada_int9_bs3/SD_SD/scheme1   
# make summarize_performance indir=output/performance2/bada_int9_bs3/SD_TD/scheme1/Final    outdir=output/summary/performance2/bada_int9_bs3/SD_TD/scheme1   
# make summarize_performance indir=output/performance2/bada_int9_bs3/SD_CUD/scheme1/Final   outdir=output/summary/performance2/bada_int9_bs3/SD_CUD/scheme1  
# make summarize_performance indir=output/performance2/bada_int9_bs3/TD_TD/scheme1/Final    outdir=output/summary/performance2/bada_int9_bs3/TD_TD/scheme1   
# make summarize_performance indir=output/performance2/bada_int9_bs3/TD_GD/scheme1/Final    outdir=output/summary/performance2/bada_int9_bs3/TD_GD/scheme1   

# make summarize_performance indir=output/performance2/bada_int9_bs3/original/scheme2/Final outdir=output/summary/performance2/bada_int9_bs3/original/scheme2
# make summarize_performance indir=output/performance2/bada_int9_bs3/Raw_SD/scheme2/Final   outdir=output/summary/performance2/bada_int9_bs3/Raw_SD/scheme2  
# make summarize_performance indir=output/performance2/bada_int9_bs3/Raw_TD/scheme2/Final   outdir=output/summary/performance2/bada_int9_bs3/Raw_TD/scheme2  
# make summarize_performance indir=output/performance2/bada_int9_bs3/Raw_CUD/scheme2/Final  outdir=output/summary/performance2/bada_int9_bs3/Raw_CUD/scheme2 
# make summarize_performance indir=output/performance2/bada_int9_bs3/SD_SD/scheme2/Final    outdir=output/summary/performance2/bada_int9_bs3/SD_SD/scheme2   
# make summarize_performance indir=output/performance2/bada_int9_bs3/SD_TD/scheme2/Final    outdir=output/summary/performance2/bada_int9_bs3/SD_TD/scheme2   
# make summarize_performance indir=output/performance2/bada_int9_bs3/SD_CUD/scheme2/Final   outdir=output/summary/performance2/bada_int9_bs3/SD_CUD/scheme2  
# make summarize_performance indir=output/performance2/bada_int9_bs3/TD_TD/scheme2/Final    outdir=output/summary/performance2/bada_int9_bs3/TD_TD/scheme2   
# make summarize_performance indir=output/performance2/bada_int9_bs3/TD_GD/scheme2/Final    outdir=output/summary/performance2/bada_int9_bs3/TD_GD/scheme2   

#* mem
# make summarize_performance indir=output/performance2/bada_int9_bs3/original/mem/scheme1/Final outdir=output/summary/performance2/bada_int9_bs3/original/mem/scheme1
# make summarize_performance indir=output/performance2/bada_int9_bs3/Raw_SD/mem/scheme1/Final   outdir=output/summary/performance2/bada_int9_bs3/Raw_SD/mem/scheme1  
# make summarize_performance indir=output/performance2/bada_int9_bs3/Raw_TD/mem/scheme1/Final   outdir=output/summary/performance2/bada_int9_bs3/Raw_TD/mem/scheme1  
# make summarize_performance indir=output/performance2/bada_int9_bs3/Raw_CUD/mem/scheme1/Final  outdir=output/summary/performance2/bada_int9_bs3/Raw_CUD/mem/scheme1 
# make summarize_performance indir=output/performance2/bada_int9_bs3/SD_SD/mem/scheme1/Final    outdir=output/summary/performance2/bada_int9_bs3/SD_SD/mem/scheme1   
# make summarize_performance indir=output/performance2/bada_int9_bs3/SD_TD/mem/scheme1/Final    outdir=output/summary/performance2/bada_int9_bs3/SD_TD/mem/scheme1   
# make summarize_performance indir=output/performance2/bada_int9_bs3/SD_CUD/mem/scheme1/Final   outdir=output/summary/performance2/bada_int9_bs3/SD_CUD/mem/scheme1  
# make summarize_performance indir=output/performance2/bada_int9_bs3/TD_TD/mem/scheme1/Final    outdir=output/summary/performance2/bada_int9_bs3/TD_TD/mem/scheme1   

# make summarize_performance indir=output/performance2/bada_int9_bs3/original/mem/scheme2/Final outdir=output/summary/performance2/bada_int9_bs3/original/mem/scheme2
# make summarize_performance indir=output/performance2/bada_int9_bs3/Raw_SD/mem/scheme2/Final   outdir=output/summary/performance2/bada_int9_bs3/Raw_SD/mem/scheme2  
# make summarize_performance indir=output/performance2/bada_int9_bs3/Raw_TD/mem/scheme2/Final   outdir=output/summary/performance2/bada_int9_bs3/Raw_TD/mem/scheme2  
# make summarize_performance indir=output/performance2/bada_int9_bs3/Raw_CUD/mem/scheme2/Final  outdir=output/summary/performance2/bada_int9_bs3/Raw_CUD/mem/scheme2 
# make summarize_performance indir=output/performance2/bada_int9_bs3/SD_SD/mem/scheme2/Final    outdir=output/summary/performance2/bada_int9_bs3/SD_SD/mem/scheme2   
# make summarize_performance indir=output/performance2/bada_int9_bs3/SD_TD/mem/scheme2/Final    outdir=output/summary/performance2/bada_int9_bs3/SD_TD/mem/scheme2   
# make summarize_performance indir=output/performance2/bada_int9_bs3/SD_CUD/mem/scheme2/Final   outdir=output/summary/performance2/bada_int9_bs3/SD_CUD/mem/scheme2  
# make summarize_performance indir=output/performance2/bada_int9_bs3/TD_TD/mem/scheme2/Final    outdir=output/summary/performance2/bada_int9_bs3/TD_TD/mem/scheme2   
# ==================================================

##! Check the include_modules and exclude_modules
#! total 114 modules hooked
#! exclude_modules = ["t_embedder", "adaLN_modulation.1"]
##! Check c_begin and c_end in Makefile





# make run_samples outdir=output/samples/config1/original n_c=1 c_begin=0 c_end=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samples/config1/Raw_SD   n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samples/config1/Raw_TD   n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samples/config1/Raw_CUD  n_c=1 c_begin=0 c_end=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samples/config1/SD_SD    n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samples/config1/SD_TD    n_c=1 c_begin=0 c_end=1 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samples/config1/SD_CUD   n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samples/config1/TD_TD    n_c=1 c_begin=0 c_end=1 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True

# make run_samples outdir=output/samples/config2/original n_c=1 c_begin=0 c_end=1 hook=True strategy=original    clamp=True bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samples/config2/Raw_SD   n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_SD      clamp=True bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samples/config2/Raw_TD   n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_TD      clamp=True bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samples/config2/Raw_CUD  n_c=1 c_begin=0 c_end=1 hook=True strategy=cfg         clamp=True bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samples/config2/SD_SD    n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial     clamp=True bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samples/config2/SD_TD    n_c=1 c_begin=0 c_end=1 hook=True strategy=SD_TD       clamp=True bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samples/config2/SD_CUD   n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial_cfg clamp=True bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samples/config2/TD_TD    n_c=1 c_begin=0 c_end=1 hook=True strategy=temporal    clamp=True bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True

# make run_samples outdir=output/samples/config3/original n_c=1 c_begin=0 c_end=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/samples/config3/Raw_SD   n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/samples/config3/Raw_TD   n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/samples/config3/Raw_CUD  n_c=1 c_begin=0 c_end=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/samples/config3/SD_SD    n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/samples/config3/SD_TD    n_c=1 c_begin=0 c_end=1 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/samples/config3/SD_CUD   n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/samples/config3/TD_TD    n_c=1 c_begin=0 c_end=1 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False

# make run_samples outdir=output/samples/config4/original n_c=1 c_begin=0 c_end=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samples/config4/Raw_SD   n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samples/config4/Raw_TD   n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samples/config4/Raw_CUD  n_c=1 c_begin=0 c_end=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samples/config4/SD_SD    n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samples/config4/SD_TD    n_c=1 c_begin=0 c_end=1 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samples/config4/SD_CUD   n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samples/config4/TD_TD    n_c=1 c_begin=0 c_end=1 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True




# 10K Samples (3s/img)
# make run_samples outdir=output/samples/original/int9 n_c=10 c_begin=0   c_end=199
# make run_samples outdir=output/samples/original/int9 n_c=10 c_begin=200 c_end=399
# make run_samples outdir=output/samples/original/int9 n_c=10 c_begin=400 c_end=599
# make run_samples outdir=output/samples/original/int9 n_c=10 c_begin=600 c_end=799
# make run_samples outdir=output/samples/original/int9 n_c=10 c_begin=800 c_end=999
# python evaluations/compress.py --sample-dir=output/samples/original/int9/samples_10K/inference --num-fid-samples=10000
# python evaluations/evaluator.py eval_batch/VIRTUAL_imagenet256_labeled.npz output/samples/original/int9/samples_10K/inference.npz

# 10K Samples (3s/img)
# make run_samples outdir=output/samples/original/int8 hook=True strategy=original clamp=True n_c=10 c_begin=0   c_end=199
# make run_samples outdir=output/samples/original/int8 hook=True strategy=original clamp=True n_c=10 c_begin=200 c_end=399
# make run_samples outdir=output/samples/original/int8 hook=True strategy=original clamp=True n_c=10 c_begin=400 c_end=599
# make run_samples outdir=output/samples/original/int8 hook=True strategy=original clamp=True n_c=10 c_begin=600 c_end=799
# make run_samples outdir=output/samples/original/int8 hook=True strategy=original clamp=True n_c=10 c_begin=800 c_end=999
# python evaluations/compress.py --sample-dir=output/samples/original/int8/samples_10K/inference --num-fid-samples=10000
# python evaluations/evaluator.py eval_batch/VIRTUAL_imagenet256_labeled.npz output/samples/original/int8/samples_10K/inference.npz

# 10K Samples (9s/img)
# make run_samples outdir=output/samples/original/bitslice_clamp hook=True strategy=original clamp=False n_c=10 c_begin=0   c_end=199 bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samples/original/bitslice_clamp hook=True strategy=original clamp=False n_c=10 c_begin=200 c_end=399 bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samples/original/bitslice_clamp hook=True strategy=original clamp=False n_c=10 c_begin=400 c_end=599 bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samples/original/bitslice_clamp hook=True strategy=original clamp=False n_c=10 c_begin=600 c_end=799 bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samples/original/bitslice_clamp hook=True strategy=original clamp=False n_c=10 c_begin=800 c_end=999 bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# python evaluations/compress.py --sample-dir=output/samples/original/bitslice_clamp/samples_10K/inference --num-fid-samples=10000
# python evaluations/evaluator.py eval_batch/VIRTUAL_imagenet256_labeled.npz output/samples/original/bitslice_clamp/samples_10K/inference.npz






# 10K Samples (16s/img)
#! Introduce Image Degradation
# make run_samples outdir=output/bitslice/bada_int9_bs3/original n_c=10 c_begin=0   c_end=199 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/original n_c=10 c_begin=200 c_end=399 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/original n_c=10 c_begin=400 c_end=599 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/original n_c=10 c_begin=600 c_end=799 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/original n_c=10 c_begin=800 c_end=999 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False

# make run_samples outdir=output/bitslice/bada_int9_bs3/Raw_SD   n_c=10 c_begin=0   c_end=199 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/Raw_SD   n_c=10 c_begin=200 c_end=399 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/Raw_SD   n_c=10 c_begin=400 c_end=599 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/Raw_SD   n_c=10 c_begin=600 c_end=799 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/Raw_SD   n_c=10 c_begin=800 c_end=999 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False

# make run_samples outdir=output/bitslice/bada_int9_bs3/Raw_TD   n_c=10 c_begin=0   c_end=199 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/Raw_TD   n_c=10 c_begin=200 c_end=399 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/Raw_TD   n_c=10 c_begin=400 c_end=599 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/Raw_TD   n_c=10 c_begin=600 c_end=799 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/Raw_TD   n_c=10 c_begin=800 c_end=999 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False

# make run_samples outdir=output/bitslice/bada_int9_bs3/Raw_CUD  n_c=10 c_begin=0   c_end=199 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/Raw_CUD  n_c=10 c_begin=200 c_end=399 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/Raw_CUD  n_c=10 c_begin=400 c_end=599 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/Raw_CUD  n_c=10 c_begin=600 c_end=799 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/Raw_CUD  n_c=10 c_begin=800 c_end=999 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False

# make run_samples outdir=output/bitslice/bada_int9_bs3/SD_SD    n_c=10 c_begin=0   c_end=199 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/SD_SD    n_c=10 c_begin=200 c_end=399 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/SD_SD    n_c=10 c_begin=400 c_end=599 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/SD_SD    n_c=10 c_begin=600 c_end=799 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/SD_SD    n_c=10 c_begin=800 c_end=999 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False

# make run_samples outdir=output/bitslice/bada_int9_bs3/SD_TD    n_c=10 c_begin=0   c_end=199 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/SD_TD    n_c=10 c_begin=200 c_end=399 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/SD_TD    n_c=10 c_begin=400 c_end=599 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/SD_TD    n_c=10 c_begin=600 c_end=799 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/SD_TD    n_c=10 c_begin=800 c_end=999 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False

# make run_samples outdir=output/bitslice/bada_int9_bs3/SD_CUD   n_c=10 c_begin=0   c_end=199 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/SD_CUD   n_c=10 c_begin=200 c_end=399 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/SD_CUD   n_c=10 c_begin=400 c_end=599 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/SD_CUD   n_c=10 c_begin=600 c_end=799 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/SD_CUD   n_c=10 c_begin=800 c_end=999 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False

# make run_samples outdir=output/bitslice/bada_int9_bs3/TD_TD    n_c=10 c_begin=0   c_end=199 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/TD_TD    n_c=10 c_begin=200 c_end=399 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/TD_TD    n_c=10 c_begin=400 c_end=599 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/TD_TD    n_c=10 c_begin=600 c_end=799 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/bitslice/bada_int9_bs3/TD_TD    n_c=10 c_begin=800 c_end=999 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False









# Compress
# python evaluations/compress.py --sample-dir=output/samples/inference --num-fid-samples=10000

# FID
# python evaluations/evaluator.py eval_batch/VIRTUAL_imagenet256_labeled.npz output/samples/inference.npz
# python evaluations/evaluator.py eval_batch/VIRTUAL_imagenet256_labeled.npz eval_batch/admnet_imagenet256.npz