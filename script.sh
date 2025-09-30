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

# BITSLICE (with bitslice_analysis.csv, 20s/img)
#! check the "Final" outdir
# make run_strategy outdir=output/bitslice/bada_int9_bs3/original n_c=1 c_begin=0 c_end=999 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_strategy outdir=output/bitslice/bada_int9_bs3/Raw_SD   n_c=1 c_begin=0 c_end=999 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_strategy outdir=output/bitslice/bada_int9_bs3/Raw_TD   n_c=1 c_begin=0 c_end=999 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_strategy outdir=output/bitslice/bada_int9_bs3/Raw_CUD  n_c=1 c_begin=0 c_end=999 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_strategy outdir=output/bitslice/bada_int9_bs3/SD_SD    n_c=1 c_begin=0 c_end=999 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_strategy outdir=output/bitslice/bada_int9_bs3/SD_TD    n_c=1 c_begin=0 c_end=999 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_strategy outdir=output/bitslice/bada_int9_bs3/SD_CUD   n_c=1 c_begin=0 c_end=999 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_strategy outdir=output/bitslice/bada_int9_bs3/TD_TD    n_c=1 c_begin=0 c_end=999 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True

# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/original/Final outdir=output/summary/bitslice/bada_int9_bs3/original bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/Raw_SD/Final   outdir=output/summary/bitslice/bada_int9_bs3/Raw_SD   bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/Raw_TD/Final   outdir=output/summary/bitslice/bada_int9_bs3/Raw_TD   bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/Raw_CUD/Final  outdir=output/summary/bitslice/bada_int9_bs3/Raw_CUD  bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/SD_SD/Final    outdir=output/summary/bitslice/bada_int9_bs3/SD_SD    bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/SD_TD/Final    outdir=output/summary/bitslice/bada_int9_bs3/SD_TD    bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/SD_CUD/Final   outdir=output/summary/bitslice/bada_int9_bs3/SD_CUD   bitslice_overflow=True
# make summarize_bitslice indir=output/bitslice/bada_int9_bs3/TD_TD/Final    outdir=output/summary/bitslice/bada_int9_bs3/TD_TD    bitslice_overflow=True

# BITWIDTH (with int9_full_distribution.csv, 60s/img)
# make run_strategy outdir=output/bitwidth/original n_c=1 c_begin=0 c_end=0 hook=True strategy=original    analysis=bitwidth
# make run_strategy outdir=output/bitwidth/Raw_SD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD      analysis=bitwidth
# make run_strategy outdir=output/bitwidth/Raw_TD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD      analysis=bitwidth
# make run_strategy outdir=output/bitwidth/Raw_CUD  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg         analysis=bitwidth
# make run_strategy outdir=output/bitwidth/SD_SD    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial     analysis=bitwidth
# make run_strategy outdir=output/bitwidth/SD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD       analysis=bitwidth
# make run_strategy outdir=output/bitwidth/SD_CUD   n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg analysis=bitwidth
# make run_strategy outdir=output/bitwidth/TD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal    analysis=bitwidth

# make summarize_bitwidth indir=output/bitwidth/original/Final outdir=output/summary/bitwidth/original
# make summarize_bitwidth indir=output/bitwidth/Raw_SD/Final   outdir=output/summary/bitwidth/Raw_SD
# make summarize_bitwidth indir=output/bitwidth/Raw_TD/Final   outdir=output/summary/bitwidth/Raw_TD
# make summarize_bitwidth indir=output/bitwidth/Raw_CUD/Final  outdir=output/summary/bitwidth/Raw_CUD
# make summarize_bitwidth indir=output/bitwidth/SD_SD/Final    outdir=output/summary/bitwidth/SD_SD
# make summarize_bitwidth indir=output/bitwidth/SD_TD/Final    outdir=output/summary/bitwidth/SD_TD
# make summarize_bitwidth indir=output/bitwidth/SD_CUD/Final   outdir=output/summary/bitwidth/SD_CUD
# make summarize_bitwidth indir=output/bitwidth/TD_TD/Final    outdir=output/summary/bitwidth/TD_TD

# Save tiles
# make run_strategy outdir=output/patterns/original n_c=1 c_begin=0 c_end=0 hook=True strategy=original   
# make run_strategy outdir=output/patterns/Raw_SD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_SD     
# make run_strategy outdir=output/patterns/Raw_TD   n_c=1 c_begin=0 c_end=0 hook=True strategy=Raw_TD     
# make run_strategy outdir=output/patterns/Raw_CUD  n_c=1 c_begin=0 c_end=0 hook=True strategy=cfg        
# make run_strategy outdir=output/patterns/SD_SD    n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial    
# make run_strategy outdir=output/patterns/SD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=SD_TD      
# make run_strategy outdir=output/patterns/SD_CUD   n_c=1 c_begin=0 c_end=0 hook=True strategy=spatial_cfg
# make run_strategy outdir=output/patterns/TD_TD    n_c=1 c_begin=0 c_end=0 hook=True strategy=temporal   

# make run_strategy outdir=output/bitwidth/original n_c=1 c_begin=0 c_end=0 hook=True strategy=original    analysis=bitwidth

##! Check the include_modules and exclude_modules
#! total 114 modules hooked
#! exclude_modules = ["t_embedder", "adaLN_modulation.1"]
##! Check c_begin and c_end in Makefile





# make run_samples outdir=output/samles/config1/original n_c=1 c_begin=0 c_end=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samles/config1/Raw_SD   n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samles/config1/Raw_TD   n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samles/config1/Raw_CUD  n_c=1 c_begin=0 c_end=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samles/config1/SD_SD    n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samles/config1/SD_TD    n_c=1 c_begin=0 c_end=1 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samles/config1/SD_CUD   n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samles/config1/TD_TD    n_c=1 c_begin=0 c_end=1 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True

# make run_samples outdir=output/samles/config2/original n_c=1 c_begin=0 c_end=1 hook=True strategy=original    clamp=True bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samles/config2/Raw_SD   n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_SD      clamp=True bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samles/config2/Raw_TD   n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_TD      clamp=True bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samles/config2/Raw_CUD  n_c=1 c_begin=0 c_end=1 hook=True strategy=cfg         clamp=True bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samles/config2/SD_SD    n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial     clamp=True bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samles/config2/SD_TD    n_c=1 c_begin=0 c_end=1 hook=True strategy=SD_TD       clamp=True bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samles/config2/SD_CUD   n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial_cfg clamp=True bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True
# make run_samples outdir=output/samles/config2/TD_TD    n_c=1 c_begin=0 c_end=1 hook=True strategy=temporal    clamp=True bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True

# make run_samples outdir=output/samles/config3/original n_c=1 c_begin=0 c_end=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/samles/config3/Raw_SD   n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/samles/config3/Raw_TD   n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/samles/config3/Raw_CUD  n_c=1 c_begin=0 c_end=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/samles/config3/SD_SD    n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/samles/config3/SD_TD    n_c=1 c_begin=0 c_end=1 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/samles/config3/SD_CUD   n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False
# make run_samples outdir=output/samles/config3/TD_TD    n_c=1 c_begin=0 c_end=1 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=False

# make run_samples outdir=output/samles/config4/original n_c=1 c_begin=0 c_end=1 hook=True strategy=original    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samles/config4/Raw_SD   n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_SD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samles/config4/Raw_TD   n_c=1 c_begin=0 c_end=1 hook=True strategy=Raw_TD      clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samles/config4/Raw_CUD  n_c=1 c_begin=0 c_end=1 hook=True strategy=cfg         clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samles/config4/SD_SD    n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial     clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samles/config4/SD_TD    n_c=1 c_begin=0 c_end=1 hook=True strategy=SD_TD       clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samles/config4/SD_CUD   n_c=1 c_begin=0 c_end=1 hook=True strategy=spatial_cfg clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True
# make run_samples outdir=output/samles/config4/TD_TD    n_c=1 c_begin=0 c_end=1 hook=True strategy=temporal    clamp=False bitslice_method=bada bitslice_bits=9 bitslice_width=3 bitslice_overflow=True bitslice_clamp=True




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