# bitwidth
# make SD n_iter=2 hook=True strategy=original                                   analysis=bitwidth
# make SD n_iter=2 hook=True strategy=spatial           ref_first=True           analysis=bitwidth
# make SD n_iter=2 hook=True strategy=spatial           ref_first=False          analysis=bitwidth
# make SD n_iter=2 hook=True strategy=temporal                                   analysis=bitwidth
# make SD n_iter=2 hook=True strategy=cfg                                        analysis=bitwidth
# make SD n_iter=2 hook=True strategy=cfg_opt                           bit_th=4 analysis=bitwidth
# make SD n_iter=2 hook=True strategy=spatial_cfg       ref_first=True           analysis=bitwidth
# make SD n_iter=2 hook=True strategy=spatial_cfg       ref_first=False          analysis=bitwidth
# make SD n_iter=2 hook=True strategy=spatial_cfg_opt   ref_first=True  bit_th=4 analysis=bitwidth
# make SD n_iter=2 hook=True strategy=spatial_cfg_opt   ref_first=False bit_th=4 analysis=bitwidth
# make SD n_iter=2 hook=True strategy=Raw_SD            ref_first=False          analysis=bitwidth
# make SD n_iter=2 hook=True strategy=Raw_TD                                     analysis=bitwidth

# similarity
# make SD n_iter=2 hook=True strategy=original ref_first=True  analysis=similarity similarity_type=spatial
# make SD n_iter=2 hook=True strategy=original ref_first=False analysis=similarity similarity_type=spatial
# make SD n_iter=2 hook=True strategy=original                 analysis=similarity similarity_type=temporal
# make SD n_iter=2 hook=True strategy=original                 analysis=similarity similarity_type=conditional

# shape
# make SD n_iter=2 hook=True analysis=shape
# ==================================================


# BITSLICE (with bitslice_analysis.csv, 65s/img)
#! check outdir="Final"
#* make SD prompt_file=prompt.txt outdir=output/test n_iter=2 hook=True strategy=original clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
#* make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt    outdir=output/coco2017/bitslice/bada_int9_bs3/original n_iter=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True

#* BADA
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/bada_int9_bs3/Raw  n_iter=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/bada_int9_bs3/SD   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/bada_int9_bs3/TD   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/bada_int9_bs3/CUD  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False

# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/bada_int9_bs3/Raw  n_iter=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/bada_int9_bs3/SD   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/bada_int9_bs3/TD   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/bada_int9_bs3/CUD  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False

# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/bada_int9_bs3/original n_iter=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/bada_int9_bs3/Raw_SD   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/bada_int9_bs3/Raw_TD   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/bada_int9_bs3/Raw_CUD  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/bada_int9_bs3/SD_SD    n_iter=1 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/bada_int9_bs3/SD_TD    n_iter=1 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/bada_int9_bs3/SD_CUD   n_iter=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/bada_int9_bs3/TD_TD    n_iter=1 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/bada_int9_bs3/TD_GD    n_iter=1 hook=True strategy=TD_GD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=False performance=False

# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/bada_int9_bs3/original n_iter=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/bada_int9_bs3/Raw_SD   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/bada_int9_bs3/Raw_TD   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/bada_int9_bs3/Raw_CUD  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/bada_int9_bs3/SD_SD    n_iter=1 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/bada_int9_bs3/SD_TD    n_iter=1 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/bada_int9_bs3/SD_CUD   n_iter=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/bada_int9_bs3/TD_TD    n_iter=1 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/bada_int9_bs3/TD_GD    n_iter=1 hook=True strategy=TD_GD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True path_reverse=True performance=False

#* Naive
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/naive_int9_bs3/Raw  n_iter=1 hook=True strategy=original    clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/naive_int9_bs3/SD   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/naive_int9_bs3/TD   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/naive_int9_bs3/CUD  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False

# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/naive_int9_bs3/Raw  n_iter=1 hook=True strategy=original    clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/naive_int9_bs3/SD   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/naive_int9_bs3/TD   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/naive_int9_bs3/CUD  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False

# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/naive_int9_bs3/original n_iter=1 hook=True strategy=original    clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/naive_int9_bs3/Raw_SD   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/naive_int9_bs3/Raw_TD   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/naive_int9_bs3/Raw_CUD  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/naive_int9_bs3/SD_SD    n_iter=1 hook=True strategy=spatial     clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/naive_int9_bs3/SD_TD    n_iter=1 hook=True strategy=SD_TD       clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/naive_int9_bs3/SD_CUD   n_iter=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/naive_int9_bs3/TD_TD    n_iter=1 hook=True strategy=temporal    clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/naive_int9_bs3/TD_GD    n_iter=1 hook=True strategy=TD_GD       clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False

# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/naive_int9_bs3/original n_iter=1 hook=True strategy=original    clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/naive_int9_bs3/Raw_SD   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/naive_int9_bs3/Raw_TD   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/naive_int9_bs3/Raw_CUD  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/naive_int9_bs3/SD_SD    n_iter=1 hook=True strategy=spatial     clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/naive_int9_bs3/SD_TD    n_iter=1 hook=True strategy=SD_TD       clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/naive_int9_bs3/SD_CUD   n_iter=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/naive_int9_bs3/TD_TD    n_iter=1 hook=True strategy=temporal    clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/naive_int9_bs3/TD_GD    n_iter=1 hook=True strategy=TD_GD       clamp=False bitslice_method=naive bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False

#* Sibia
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/sibia_int9_bs3/Raw  n_iter=1 hook=True strategy=original    clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/sibia_int9_bs3/SD   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/sibia_int9_bs3/TD   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/sibia_int9_bs3/CUD  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=3 path_reverse=False performance=False

# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/sibia_int9_bs3/Raw  n_iter=1 hook=True strategy=original    clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/sibia_int9_bs3/SD   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/sibia_int9_bs3/TD   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/sibia_int9_bs3/CUD  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=3 path_reverse=True performance=False

# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/sibia_int9_bs3/original n_iter=1 hook=True strategy=original    clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/sibia_int9_bs3/Raw_SD   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/sibia_int9_bs3/Raw_TD   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/sibia_int9_bs3/Raw_CUD  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/sibia_int9_bs3/SD_SD    n_iter=1 hook=True strategy=spatial     clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/sibia_int9_bs3/SD_TD    n_iter=1 hook=True strategy=SD_TD       clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/sibia_int9_bs3/SD_CUD   n_iter=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/sibia_int9_bs3/TD_TD    n_iter=1 hook=True strategy=temporal    clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice/sibia_int9_bs3/TD_GD    n_iter=1 hook=True strategy=TD_GD       clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=False performance=False

# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/sibia_int9_bs3/original n_iter=1 hook=True strategy=original    clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/sibia_int9_bs3/Raw_SD   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/sibia_int9_bs3/Raw_TD   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/sibia_int9_bs3/Raw_CUD  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/sibia_int9_bs3/SD_SD    n_iter=1 hook=True strategy=spatial     clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/sibia_int9_bs3/SD_TD    n_iter=1 hook=True strategy=SD_TD       clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/sibia_int9_bs3/SD_CUD   n_iter=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/sibia_int9_bs3/TD_TD    n_iter=1 hook=True strategy=temporal    clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitslice2/sibia_int9_bs3/TD_GD    n_iter=1 hook=True strategy=TD_GD       clamp=False bitslice_method=sibia bitslice_bits=9 bitslice_width=5 path_reverse=True performance=False




# ==================================================
# BITSLICE SUMMARY
#* BADA
# make summarize_bitslice indir=output/coco2017/bitslice/bada_int9_bs3/Raw/Final  outdir=output/coco2017/summary/bitslice/bada_int9_bs3/Raw  bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice/bada_int9_bs3/SD/Final   outdir=output/coco2017/summary/bitslice/bada_int9_bs3/SD   bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice/bada_int9_bs3/TD/Final   outdir=output/coco2017/summary/bitslice/bada_int9_bs3/TD   bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice/bada_int9_bs3/CUD/Final  outdir=output/coco2017/summary/bitslice/bada_int9_bs3/CUD  bitslice_overflow=True

# make summarize_bitslice indir=output/coco2017/bitslice2/bada_int9_bs3/Raw/Final  outdir=output/coco2017/summary/bitslice2/bada_int9_bs3/Raw  bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice2/bada_int9_bs3/SD/Final   outdir=output/coco2017/summary/bitslice2/bada_int9_bs3/SD   bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice2/bada_int9_bs3/TD/Final   outdir=output/coco2017/summary/bitslice2/bada_int9_bs3/TD   bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice2/bada_int9_bs3/CUD/Final  outdir=output/coco2017/summary/bitslice2/bada_int9_bs3/CUD  bitslice_overflow=True

# make summarize_bitslice indir=output/coco2017/bitslice/bada_int9_bs3/original/Final outdir=output/coco2017/summary/bitslice/bada_int9_bs3/original bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice/bada_int9_bs3/Raw_SD/Final   outdir=output/coco2017/summary/bitslice/bada_int9_bs3/Raw_SD   bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice/bada_int9_bs3/Raw_TD/Final   outdir=output/coco2017/summary/bitslice/bada_int9_bs3/Raw_TD   bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice/bada_int9_bs3/Raw_CUD/Final  outdir=output/coco2017/summary/bitslice/bada_int9_bs3/Raw_CUD  bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice/bada_int9_bs3/SD_SD/Final    outdir=output/coco2017/summary/bitslice/bada_int9_bs3/SD_SD    bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice/bada_int9_bs3/SD_TD/Final    outdir=output/coco2017/summary/bitslice/bada_int9_bs3/SD_TD    bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice/bada_int9_bs3/SD_CUD/Final   outdir=output/coco2017/summary/bitslice/bada_int9_bs3/SD_CUD   bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice/bada_int9_bs3/TD_TD/Final    outdir=output/coco2017/summary/bitslice/bada_int9_bs3/TD_TD    bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice/bada_int9_bs3/TD_GD/Final    outdir=output/coco2017/summary/bitslice/bada_int9_bs3/TD_GD    bitslice_overflow=True

# make summarize_bitslice indir=output/coco2017/bitslice2/bada_int9_bs3/original/Final outdir=output/coco2017/summary/bitslice2/bada_int9_bs3/original bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice2/bada_int9_bs3/Raw_SD/Final   outdir=output/coco2017/summary/bitslice2/bada_int9_bs3/Raw_SD   bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice2/bada_int9_bs3/Raw_TD/Final   outdir=output/coco2017/summary/bitslice2/bada_int9_bs3/Raw_TD   bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice2/bada_int9_bs3/Raw_CUD/Final  outdir=output/coco2017/summary/bitslice2/bada_int9_bs3/Raw_CUD  bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice2/bada_int9_bs3/SD_SD/Final    outdir=output/coco2017/summary/bitslice2/bada_int9_bs3/SD_SD    bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice2/bada_int9_bs3/SD_TD/Final    outdir=output/coco2017/summary/bitslice2/bada_int9_bs3/SD_TD    bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice2/bada_int9_bs3/SD_CUD/Final   outdir=output/coco2017/summary/bitslice2/bada_int9_bs3/SD_CUD   bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice2/bada_int9_bs3/TD_TD/Final    outdir=output/coco2017/summary/bitslice2/bada_int9_bs3/TD_TD    bitslice_overflow=True
# make summarize_bitslice indir=output/coco2017/bitslice2/bada_int9_bs3/TD_GD/Final    outdir=output/coco2017/summary/bitslice2/bada_int9_bs3/TD_GD    bitslice_overflow=True

#* Naive
# make summarize_bitslice indir=output/coco2017/bitslice/naive_int9_bs3/Raw/Final  outdir=output/coco2017/summary/bitslice/naive_int9_bs3/Raw  bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/naive_int9_bs3/SD/Final   outdir=output/coco2017/summary/bitslice/naive_int9_bs3/SD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/naive_int9_bs3/TD/Final   outdir=output/coco2017/summary/bitslice/naive_int9_bs3/TD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/naive_int9_bs3/CUD/Final  outdir=output/coco2017/summary/bitslice/naive_int9_bs3/CUD  bitslice_overflow=False

# make summarize_bitslice indir=output/coco2017/bitslice2/naive_int9_bs3/Raw/Final  outdir=output/coco2017/summary/bitslice2/naive_int9_bs3/Raw  bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/naive_int9_bs3/SD/Final   outdir=output/coco2017/summary/bitslice2/naive_int9_bs3/SD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/naive_int9_bs3/TD/Final   outdir=output/coco2017/summary/bitslice2/naive_int9_bs3/TD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/naive_int9_bs3/CUD/Final  outdir=output/coco2017/summary/bitslice2/naive_int9_bs3/CUD  bitslice_overflow=False

# make summarize_bitslice indir=output/coco2017/bitslice/naive_int9_bs3/original/Final outdir=output/coco2017/summary/bitslice/naive_int9_bs3/original bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/naive_int9_bs3/Raw_SD/Final   outdir=output/coco2017/summary/bitslice/naive_int9_bs3/Raw_SD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/naive_int9_bs3/Raw_TD/Final   outdir=output/coco2017/summary/bitslice/naive_int9_bs3/Raw_TD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/naive_int9_bs3/Raw_CUD/Final  outdir=output/coco2017/summary/bitslice/naive_int9_bs3/Raw_CUD  bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/naive_int9_bs3/SD_SD/Final    outdir=output/coco2017/summary/bitslice/naive_int9_bs3/SD_SD    bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/naive_int9_bs3/SD_TD/Final    outdir=output/coco2017/summary/bitslice/naive_int9_bs3/SD_TD    bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/naive_int9_bs3/SD_CUD/Final   outdir=output/coco2017/summary/bitslice/naive_int9_bs3/SD_CUD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/naive_int9_bs3/TD_TD/Final    outdir=output/coco2017/summary/bitslice/naive_int9_bs3/TD_TD    bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/naive_int9_bs3/TD_GD/Final    outdir=output/coco2017/summary/bitslice/naive_int9_bs3/TD_GD    bitslice_overflow=False

# make summarize_bitslice indir=output/coco2017/bitslice2/naive_int9_bs3/original/Final outdir=output/coco2017/summary/bitslice2/naive_int9_bs3/original bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/naive_int9_bs3/Raw_SD/Final   outdir=output/coco2017/summary/bitslice2/naive_int9_bs3/Raw_SD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/naive_int9_bs3/Raw_TD/Final   outdir=output/coco2017/summary/bitslice2/naive_int9_bs3/Raw_TD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/naive_int9_bs3/Raw_CUD/Final  outdir=output/coco2017/summary/bitslice2/naive_int9_bs3/Raw_CUD  bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/naive_int9_bs3/SD_SD/Final    outdir=output/coco2017/summary/bitslice2/naive_int9_bs3/SD_SD    bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/naive_int9_bs3/SD_TD/Final    outdir=output/coco2017/summary/bitslice2/naive_int9_bs3/SD_TD    bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/naive_int9_bs3/SD_CUD/Final   outdir=output/coco2017/summary/bitslice2/naive_int9_bs3/SD_CUD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/naive_int9_bs3/TD_TD/Final    outdir=output/coco2017/summary/bitslice2/naive_int9_bs3/TD_TD    bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/naive_int9_bs3/TD_GD/Final    outdir=output/coco2017/summary/bitslice2/naive_int9_bs3/TD_GD    bitslice_overflow=False

#* Sibia
# make summarize_bitslice indir=output/coco2017/bitslice/sibia_int9_bs3/Raw/Final  outdir=output/coco2017/summary/bitslice/sibia_int9_bs3/Raw  bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/sibia_int9_bs3/SD/Final   outdir=output/coco2017/summary/bitslice/sibia_int9_bs3/SD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/sibia_int9_bs3/TD/Final   outdir=output/coco2017/summary/bitslice/sibia_int9_bs3/TD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/sibia_int9_bs3/CUD/Final  outdir=output/coco2017/summary/bitslice/sibia_int9_bs3/CUD  bitslice_overflow=False

# make summarize_bitslice indir=output/coco2017/bitslice2/sibia_int9_bs3/Raw/Final  outdir=output/coco2017/summary/bitslice2/sibia_int9_bs3/Raw  bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/sibia_int9_bs3/SD/Final   outdir=output/coco2017/summary/bitslice2/sibia_int9_bs3/SD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/sibia_int9_bs3/TD/Final   outdir=output/coco2017/summary/bitslice2/sibia_int9_bs3/TD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/sibia_int9_bs3/CUD/Final  outdir=output/coco2017/summary/bitslice2/sibia_int9_bs3/CUD  bitslice_overflow=False

# make summarize_bitslice indir=output/coco2017/bitslice/sibia_int9_bs3/original/Final outdir=output/coco2017/summary/bitslice/sibia_int9_bs3/original bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/sibia_int9_bs3/Raw_SD/Final   outdir=output/coco2017/summary/bitslice/sibia_int9_bs3/Raw_SD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/sibia_int9_bs3/Raw_TD/Final   outdir=output/coco2017/summary/bitslice/sibia_int9_bs3/Raw_TD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/sibia_int9_bs3/Raw_CUD/Final  outdir=output/coco2017/summary/bitslice/sibia_int9_bs3/Raw_CUD  bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/sibia_int9_bs3/SD_SD/Final    outdir=output/coco2017/summary/bitslice/sibia_int9_bs3/SD_SD    bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/sibia_int9_bs3/SD_TD/Final    outdir=output/coco2017/summary/bitslice/sibia_int9_bs3/SD_TD    bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/sibia_int9_bs3/SD_CUD/Final   outdir=output/coco2017/summary/bitslice/sibia_int9_bs3/SD_CUD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/sibia_int9_bs3/TD_TD/Final    outdir=output/coco2017/summary/bitslice/sibia_int9_bs3/TD_TD    bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice/sibia_int9_bs3/TD_GD/Final    outdir=output/coco2017/summary/bitslice/sibia_int9_bs3/TD_GD    bitslice_overflow=False

# make summarize_bitslice indir=output/coco2017/bitslice2/sibia_int9_bs3/original/Final outdir=output/coco2017/summary/bitslice2/sibia_int9_bs3/original bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/sibia_int9_bs3/Raw_SD/Final   outdir=output/coco2017/summary/bitslice2/sibia_int9_bs3/Raw_SD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/sibia_int9_bs3/Raw_TD/Final   outdir=output/coco2017/summary/bitslice2/sibia_int9_bs3/Raw_TD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/sibia_int9_bs3/Raw_CUD/Final  outdir=output/coco2017/summary/bitslice2/sibia_int9_bs3/Raw_CUD  bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/sibia_int9_bs3/SD_SD/Final    outdir=output/coco2017/summary/bitslice2/sibia_int9_bs3/SD_SD    bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/sibia_int9_bs3/SD_TD/Final    outdir=output/coco2017/summary/bitslice2/sibia_int9_bs3/SD_TD    bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/sibia_int9_bs3/SD_CUD/Final   outdir=output/coco2017/summary/bitslice2/sibia_int9_bs3/SD_CUD   bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/sibia_int9_bs3/TD_TD/Final    outdir=output/coco2017/summary/bitslice2/sibia_int9_bs3/TD_TD    bitslice_overflow=False
# make summarize_bitslice indir=output/coco2017/bitslice2/sibia_int9_bs3/TD_GD/Final    outdir=output/coco2017/summary/bitslice2/sibia_int9_bs3/TD_GD    bitslice_overflow=False



# ==================================================


# BITWIDTH (with int9_full_distribution.csv, #2.5m/img)
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitwidth/Raw      n_iter=1 hook=True strategy=original    analysis=bitwidth path_reverse=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitwidth/SD       n_iter=1 hook=True strategy=Raw_SD      analysis=bitwidth path_reverse=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitwidth/TD       n_iter=1 hook=True strategy=Raw_TD      analysis=bitwidth path_reverse=False
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitwidth/CUD      n_iter=1 hook=True strategy=cfg         analysis=bitwidth path_reverse=False

# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitwidth2/Raw      n_iter=1 hook=True strategy=original    analysis=bitwidth path_reverse=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitwidth2/SD       n_iter=1 hook=True strategy=Raw_SD      analysis=bitwidth path_reverse=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitwidth2/TD       n_iter=1 hook=True strategy=Raw_TD      analysis=bitwidth path_reverse=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitwidth2/CUD      n_iter=1 hook=True strategy=cfg         analysis=bitwidth path_reverse=True

# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitwidth/original n_iter=1 hook=True strategy=original    analysis=bitwidth
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitwidth/Raw_SD   n_iter=1 hook=True strategy=Raw_SD      analysis=bitwidth
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitwidth/Raw_TD   n_iter=1 hook=True strategy=Raw_TD      analysis=bitwidth
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitwidth/Raw_CUD  n_iter=1 hook=True strategy=cfg         analysis=bitwidth
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitwidth/SD_SD    n_iter=1 hook=True strategy=spatial     analysis=bitwidth
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitwidth/SD_TD    n_iter=1 hook=True strategy=SD_TD       analysis=bitwidth
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitwidth/SD_CUD   n_iter=1 hook=True strategy=spatial_cfg analysis=bitwidth
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/bitwidth/TD_TD    n_iter=1 hook=True strategy=temporal    analysis=bitwidth
# ==================================================


# BITWIDTH SUMMARY
# make summarize_bitwidth indir=output/coco2017/bitwidth/Raw/Final      outdir=output/coco2017/summary/bitwidth/Raw
# make summarize_bitwidth indir=output/coco2017/bitwidth/SD/Final       outdir=output/coco2017/summary/bitwidth/SD
# make summarize_bitwidth indir=output/coco2017/bitwidth/TD/Final       outdir=output/coco2017/summary/bitwidth/TD
# make summarize_bitwidth indir=output/coco2017/bitwidth/CUD/Final      outdir=output/coco2017/summary/bitwidth/CUD

# make summarize_bitwidth indir=output/coco2017/bitwidth2/Raw/Final      outdir=output/coco2017/summary/bitwidth2/Raw
# make summarize_bitwidth indir=output/coco2017/bitwidth2/SD/Final       outdir=output/coco2017/summary/bitwidth2/SD
# make summarize_bitwidth indir=output/coco2017/bitwidth2/TD/Final       outdir=output/coco2017/summary/bitwidth2/TD
# make summarize_bitwidth indir=output/coco2017/bitwidth2/CUD/Final      outdir=output/coco2017/summary/bitwidth2/CUD

# make summarize_bitwidth indir=output/coco2017/bitwidth/original/Final outdir=output/coco2017/summary/bitwidth/original
# make summarize_bitwidth indir=output/coco2017/bitwidth/Raw_SD/Final   outdir=output/coco2017/summary/bitwidth/Raw_SD
# make summarize_bitwidth indir=output/coco2017/bitwidth/Raw_TD/Final   outdir=output/coco2017/summary/bitwidth/Raw_TD
# make summarize_bitwidth indir=output/coco2017/bitwidth/Raw_CUD/Final  outdir=output/coco2017/summary/bitwidth/Raw_CUD
# make summarize_bitwidth indir=output/coco2017/bitwidth/SD_SD/Final    outdir=output/coco2017/summary/bitwidth/SD_SD
# make summarize_bitwidth indir=output/coco2017/bitwidth/SD_TD/Final    outdir=output/coco2017/summary/bitwidth/SD_TD
# make summarize_bitwidth indir=output/coco2017/bitwidth/SD_CUD/Final   outdir=output/coco2017/summary/bitwidth/SD_CUD
# make summarize_bitwidth indir=output/coco2017/bitwidth/TD_TD/Final    outdir=output/coco2017/summary/bitwidth/TD_TD
# ==================================================


# TILE SAVING
#! check save="True"
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/patterns/original n_iter=1 hook=True strategy=original   
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/patterns/Raw_SD   n_iter=1 hook=True strategy=Raw_SD     
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/patterns/Raw_TD   n_iter=1 hook=True strategy=Raw_TD     
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/patterns/Raw_CUD  n_iter=1 hook=True strategy=cfg        
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/patterns/SD_SD    n_iter=1 hook=True strategy=spatial    
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/patterns/SD_TD    n_iter=1 hook=True strategy=SD_TD      
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/patterns/SD_CUD   n_iter=1 hook=True strategy=spatial_cfg
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/patterns/TD_TD    n_iter=1 hook=True strategy=temporal   
# ==================================================


# PERFORMANCE (with performance_analysis.csv, 10m/img)
#! check scheme in bitslice_analyzer.py, and no normal bitslice anlayze (in create_strategy)
#* PATH_REVERSE=FALSE
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/original/scheme1 n_iter=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/Raw_SD/scheme1   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/Raw_TD/scheme1   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/Raw_CUD/scheme1  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/SD_SD/scheme1    n_iter=1 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/SD_TD/scheme1    n_iter=1 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/SD_CUD/scheme1   n_iter=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/TD_TD/scheme1    n_iter=1 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/TD_GD/scheme1    n_iter=1 hook=True strategy=TD_GD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True

# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/original/scheme2 n_iter=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/Raw_SD/scheme2   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/Raw_TD/scheme2   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/Raw_CUD/scheme2  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/SD_SD/scheme2    n_iter=1 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/SD_TD/scheme2    n_iter=1 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/SD_CUD/scheme2   n_iter=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/TD_TD/scheme2    n_iter=1 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/TD_GD/scheme2    n_iter=1 hook=True strategy=TD_GD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True

#* mem
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/original/mem/scheme1 n_iter=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/Raw_SD/mem/scheme1   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/Raw_TD/mem/scheme1   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/Raw_CUD/mem/scheme1  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/SD_SD/mem/scheme1    n_iter=1 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/SD_TD/mem/scheme1    n_iter=1 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/SD_CUD/mem/scheme1   n_iter=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/TD_TD/mem/scheme1    n_iter=1 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=False performance=True

# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/original/mem/scheme2 n_iter=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/Raw_SD/mem/scheme2   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/Raw_TD/mem/scheme2   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/Raw_CUD/mem/scheme2  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/SD_SD/mem/scheme2    n_iter=1 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/SD_TD/mem/scheme2    n_iter=1 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/SD_CUD/mem/scheme2   n_iter=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance/bada_int9_bs3/TD_TD/mem/scheme2    n_iter=1 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=False performance=True

#* PATH_REVERSE=TRUE
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/original/scheme1 n_iter=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/Raw_SD/scheme1   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/Raw_TD/scheme1   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/Raw_CUD/scheme1  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/SD_SD/scheme1    n_iter=1 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/SD_TD/scheme1    n_iter=1 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/SD_CUD/scheme1   n_iter=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/TD_TD/scheme1    n_iter=1 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/TD_GD/scheme1    n_iter=1 hook=True strategy=TD_GD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True

# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/original/scheme2 n_iter=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/Raw_SD/scheme2   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/Raw_TD/scheme2   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/Raw_CUD/scheme2  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/SD_SD/scheme2    n_iter=1 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/SD_TD/scheme2    n_iter=1 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/SD_CUD/scheme2   n_iter=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/TD_TD/scheme2    n_iter=1 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/TD_GD/scheme2    n_iter=1 hook=True strategy=TD_GD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True

#* mem
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/original/mem/scheme1 n_iter=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/Raw_SD/mem/scheme1   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/Raw_TD/mem/scheme1   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/Raw_CUD/mem/scheme1  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/SD_SD/mem/scheme1    n_iter=1 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/SD_TD/mem/scheme1    n_iter=1 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/SD_CUD/mem/scheme1   n_iter=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/TD_TD/mem/scheme1    n_iter=1 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=1 path_reverse=True performance=True

# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/original/mem/scheme2 n_iter=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/Raw_SD/mem/scheme2   n_iter=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/Raw_TD/mem/scheme2   n_iter=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/Raw_CUD/mem/scheme2  n_iter=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/SD_SD/mem/scheme2    n_iter=1 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/SD_TD/mem/scheme2    n_iter=1 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/SD_CUD/mem/scheme2   n_iter=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# make SD prompt_file=/work/cwhsu0720/dataset/coco2017/coco2017_val_1.txt outdir=output/coco2017/performance2/bada_int9_bs3/TD_TD/mem/scheme2    n_iter=1 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True scheme=2 path_reverse=True performance=True
# ==================================================


# PERFORMANCE SUMMARY
#* PATH_REVERSE=FALSE, SCHEME1, no memory transfer latency
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/original/scheme1/Final outdir=output/coco2017/summary/performance/bada_int9_bs3/original/scheme1
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/Raw_SD/scheme1/Final   outdir=output/coco2017/summary/performance/bada_int9_bs3/Raw_SD/scheme1  
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/Raw_TD/scheme1/Final   outdir=output/coco2017/summary/performance/bada_int9_bs3/Raw_TD/scheme1  
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/Raw_CUD/scheme1/Final  outdir=output/coco2017/summary/performance/bada_int9_bs3/Raw_CUD/scheme1 
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/SD_SD/scheme1/Final    outdir=output/coco2017/summary/performance/bada_int9_bs3/SD_SD/scheme1   
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/SD_TD/scheme1/Final    outdir=output/coco2017/summary/performance/bada_int9_bs3/SD_TD/scheme1   
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/SD_CUD/scheme1/Final   outdir=output/coco2017/summary/performance/bada_int9_bs3/SD_CUD/scheme1  
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/TD_TD/scheme1/Final    outdir=output/coco2017/summary/performance/bada_int9_bs3/TD_TD/scheme1   
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/TD_GD/scheme1/Final    outdir=output/coco2017/summary/performance/bada_int9_bs3/TD_GD/scheme1   

#* PATH_REVERSE=FALSE, SCHEME2, no memory transfer latency
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/original/scheme2/Final outdir=output/coco2017/summary/performance/bada_int9_bs3/original/scheme2
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/Raw_SD/scheme2/Final   outdir=output/coco2017/summary/performance/bada_int9_bs3/Raw_SD/scheme2  
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/Raw_TD/scheme2/Final   outdir=output/coco2017/summary/performance/bada_int9_bs3/Raw_TD/scheme2  
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/Raw_CUD/scheme2/Final  outdir=output/coco2017/summary/performance/bada_int9_bs3/Raw_CUD/scheme2 
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/SD_SD/scheme2/Final    outdir=output/coco2017/summary/performance/bada_int9_bs3/SD_SD/scheme2   
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/SD_TD/scheme2/Final    outdir=output/coco2017/summary/performance/bada_int9_bs3/SD_TD/scheme2   
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/SD_CUD/scheme2/Final   outdir=output/coco2017/summary/performance/bada_int9_bs3/SD_CUD/scheme2  
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/TD_TD/scheme2/Final    outdir=output/coco2017/summary/performance/bada_int9_bs3/TD_TD/scheme2   
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/TD_GD/scheme2/Final    outdir=output/coco2017/summary/performance/bada_int9_bs3/TD_GD/scheme2   

#* mem
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/original/mem/scheme1/Final outdir=output/coco2017/summary/performance/bada_int9_bs3/original/mem/scheme1
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/Raw_SD/mem/scheme1/Final   outdir=output/coco2017/summary/performance/bada_int9_bs3/Raw_SD/mem/scheme1  
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/Raw_TD/mem/scheme1/Final   outdir=output/coco2017/summary/performance/bada_int9_bs3/Raw_TD/mem/scheme1  
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/Raw_CUD/mem/scheme1/Final  outdir=output/coco2017/summary/performance/bada_int9_bs3/Raw_CUD/mem/scheme1 
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/SD_SD/mem/scheme1/Final    outdir=output/coco2017/summary/performance/bada_int9_bs3/SD_SD/mem/scheme1   
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/SD_TD/mem/scheme1/Final    outdir=output/coco2017/summary/performance/bada_int9_bs3/SD_TD/mem/scheme1   
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/SD_CUD/mem/scheme1/Final   outdir=output/coco2017/summary/performance/bada_int9_bs3/SD_CUD/mem/scheme1  
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/TD_TD/mem/scheme1/Final    outdir=output/coco2017/summary/performance/bada_int9_bs3/TD_TD/mem/scheme1   

# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/original/mem/scheme2/Final outdir=output/coco2017/summary/performance/bada_int9_bs3/original/mem/scheme2
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/Raw_SD/mem/scheme2/Final   outdir=output/coco2017/summary/performance/bada_int9_bs3/Raw_SD/mem/scheme2  
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/Raw_TD/mem/scheme2/Final   outdir=output/coco2017/summary/performance/bada_int9_bs3/Raw_TD/mem/scheme2  
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/Raw_CUD/mem/scheme2/Final  outdir=output/coco2017/summary/performance/bada_int9_bs3/Raw_CUD/mem/scheme2 
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/SD_SD/mem/scheme2/Final    outdir=output/coco2017/summary/performance/bada_int9_bs3/SD_SD/mem/scheme2   
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/SD_TD/mem/scheme2/Final    outdir=output/coco2017/summary/performance/bada_int9_bs3/SD_TD/mem/scheme2   
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/SD_CUD/mem/scheme2/Final   outdir=output/coco2017/summary/performance/bada_int9_bs3/SD_CUD/mem/scheme2  
# make summarize_performance indir=output/coco2017/performance/bada_int9_bs3/TD_TD/mem/scheme2/Final    outdir=output/coco2017/summary/performance/bada_int9_bs3/TD_TD/mem/scheme2   

#* PATH_REVERSE=TRUE, SCHEME1, no memory transfer latency
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/original/scheme1/Final outdir=output/coco2017/summary/performance2/bada_int9_bs3/original/scheme1
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/Raw_SD/scheme1/Final   outdir=output/coco2017/summary/performance2/bada_int9_bs3/Raw_SD/scheme1  
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/Raw_TD/scheme1/Final   outdir=output/coco2017/summary/performance2/bada_int9_bs3/Raw_TD/scheme1  
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/Raw_CUD/scheme1/Final  outdir=output/coco2017/summary/performance2/bada_int9_bs3/Raw_CUD/scheme1 
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/SD_SD/scheme1/Final    outdir=output/coco2017/summary/performance2/bada_int9_bs3/SD_SD/scheme1   
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/SD_TD/scheme1/Final    outdir=output/coco2017/summary/performance2/bada_int9_bs3/SD_TD/scheme1   
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/SD_CUD/scheme1/Final   outdir=output/coco2017/summary/performance2/bada_int9_bs3/SD_CUD/scheme1  
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/TD_TD/scheme1/Final    outdir=output/coco2017/summary/performance2/bada_int9_bs3/TD_TD/scheme1   
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/TD_GD/scheme1/Final    outdir=output/coco2017/summary/performance2/bada_int9_bs3/TD_GD/scheme1   

#* PATH_REVERSE=TRUE, SCHEME2, no memory transfer latency
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/original/scheme2/Final outdir=output/coco2017/summary/performance2/bada_int9_bs3/original/scheme2
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/Raw_SD/scheme2/Final   outdir=output/coco2017/summary/performance2/bada_int9_bs3/Raw_SD/scheme2  
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/Raw_TD/scheme2/Final   outdir=output/coco2017/summary/performance2/bada_int9_bs3/Raw_TD/scheme2  
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/Raw_CUD/scheme2/Final  outdir=output/coco2017/summary/performance2/bada_int9_bs3/Raw_CUD/scheme2 
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/SD_SD/scheme2/Final    outdir=output/coco2017/summary/performance2/bada_int9_bs3/SD_SD/scheme2   
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/SD_TD/scheme2/Final    outdir=output/coco2017/summary/performance2/bada_int9_bs3/SD_TD/scheme2   
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/SD_CUD/scheme2/Final   outdir=output/coco2017/summary/performance2/bada_int9_bs3/SD_CUD/scheme2  
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/TD_TD/scheme2/Final    outdir=output/coco2017/summary/performance2/bada_int9_bs3/TD_TD/scheme2   
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/TD_GD/scheme2/Final    outdir=output/coco2017/summary/performance2/bada_int9_bs3/TD_GD/scheme2   

#* mem
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/original/mem/scheme1/Final outdir=output/coco2017/summary/performance2/bada_int9_bs3/original/mem/scheme1
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/Raw_SD/mem/scheme1/Final   outdir=output/coco2017/summary/performance2/bada_int9_bs3/Raw_SD/mem/scheme1  
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/Raw_TD/mem/scheme1/Final   outdir=output/coco2017/summary/performance2/bada_int9_bs3/Raw_TD/mem/scheme1  
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/Raw_CUD/mem/scheme1/Final  outdir=output/coco2017/summary/performance2/bada_int9_bs3/Raw_CUD/mem/scheme1 
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/SD_SD/mem/scheme1/Final    outdir=output/coco2017/summary/performance2/bada_int9_bs3/SD_SD/mem/scheme1   
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/SD_TD/mem/scheme1/Final    outdir=output/coco2017/summary/performance2/bada_int9_bs3/SD_TD/mem/scheme1   
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/SD_CUD/mem/scheme1/Final   outdir=output/coco2017/summary/performance2/bada_int9_bs3/SD_CUD/mem/scheme1  
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/TD_TD/mem/scheme1/Final    outdir=output/coco2017/summary/performance2/bada_int9_bs3/TD_TD/mem/scheme1   

# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/original/mem/scheme2/Final outdir=output/coco2017/summary/performance2/bada_int9_bs3/original/mem/scheme2
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/Raw_SD/mem/scheme2/Final   outdir=output/coco2017/summary/performance2/bada_int9_bs3/Raw_SD/mem/scheme2  
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/Raw_TD/mem/scheme2/Final   outdir=output/coco2017/summary/performance2/bada_int9_bs3/Raw_TD/mem/scheme2  
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/Raw_CUD/mem/scheme2/Final  outdir=output/coco2017/summary/performance2/bada_int9_bs3/Raw_CUD/mem/scheme2 
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/SD_SD/mem/scheme2/Final    outdir=output/coco2017/summary/performance2/bada_int9_bs3/SD_SD/mem/scheme2   
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/SD_TD/mem/scheme2/Final    outdir=output/coco2017/summary/performance2/bada_int9_bs3/SD_TD/mem/scheme2   
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/SD_CUD/mem/scheme2/Final   outdir=output/coco2017/summary/performance2/bada_int9_bs3/SD_CUD/mem/scheme2  
# make summarize_performance indir=output/coco2017/performance2/bada_int9_bs3/TD_TD/mem/scheme2/Final    outdir=output/coco2017/summary/performance2/bada_int9_bs3/TD_TD/mem/scheme2   
# ==================================================


##! Check the include_modules and exclude_modules
#! total 258 modules hooked
#! exclude_modules = ["time_embed.0", "time_embed.2", "emb_layers.1"]
##! Check n_iter in Makefile
