import argparse, os, datetime, gc, yaml
import logging
import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
import torch
import torch.nn as nn
from torch import autocast
from contextlib import nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from qdiff import (
    QuantModel, QuantModule, BaseQuantBlock, 
    block_reconstruction, layer_reconstruction,
)
from qdiff.adaptive_rounding import AdaRoundQuantizer
from qdiff.quant_layer import UniformAffineQuantizer
from qdiff.utils import resume_cali_model, get_train_samples
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

logger = logging.getLogger(__name__)

# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    logging.info(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        logging.info(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        logging.info("missing keys:")
        logging.info(m)
    if len(u) > 0 and verbose:
        logging.info("unexpected keys:")
        logging.info(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    # linear quantization configs
    parser.add_argument(
        "--ptq", action="store_true", help="apply post-training quantization"
    )
    parser.add_argument(
        "--quant_act", action="store_true", 
        help="if to quantize activations when ptq==True"
    )
    parser.add_argument(
        "--weight_bit",
        type=int,
        default=8,
        help="int bit for weight quantization",
    )
    parser.add_argument(
        "--act_bit",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--quant_mode", type=str, default="symmetric", 
        choices=["linear", "squant", "qdiff"], 
        help="quantization mode to use"
    )

    # qdiff specific configs
    parser.add_argument(
        "--cali_st", type=int, default=1, 
        help="number of timesteps used for calibration"
    )
    parser.add_argument(
        "--cali_batch_size", type=int, default=32, 
        help="batch size for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_n", type=int, default=1024, 
        help="number of samples for each timestep for qdiff reconstruction"
    )
    parser.add_argument(
        "--cali_iters", type=int, default=20000, 
        help="number of iterations for each qdiff reconstruction"
    )
    parser.add_argument('--cali_iters_a', default=5000, type=int, 
        help='number of iteration for LSQ')
    parser.add_argument('--cali_lr', default=4e-4, type=float, 
        help='learning rate for LSQ')
    parser.add_argument('--cali_p', default=2.4, type=float, 
        help='L_p norm minimization for LSQ')
    parser.add_argument(
        "--cali_ckpt", type=str,
        help="path for calibrated model ckpt"
    )
    parser.add_argument(
        "--cali_data_path", type=str, default="sd_coco_sample1024_allst.pt",
        help="calibration dataset name"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="resume the calibrated qdiff model"
    )
    parser.add_argument(
        "--resume_w", action="store_true",
        help="resume the calibrated qdiff model weights only"
    )
    parser.add_argument(
        "--cond", action="store_true",
        help="whether to use conditional guidance"
    )
    parser.add_argument(
        "--no_grad_ckpt", action="store_true",
        help="disable gradient checkpointing"
    )
    parser.add_argument(
        "--split", action="store_true",
        help="use split strategy in skip connection"
    )
    parser.add_argument(
        "--running_stat", action="store_true",
        help="use running statistics for act quantizers"
    )
    parser.add_argument(
        "--rs_sm_only", action="store_true",
        help="use running statistics only for softmax act quantizers"
    )
    parser.add_argument(
        "--sm_abit",type=int, default=8,
        help="attn softmax activation bit"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    ##################################################
    parser.add_argument("--hook", action="store_true", default=False, help="Hook layers")
    parser.add_argument("--strategy", type=str, choices=["original", "spatial", "temporal", "cfg", "spatial_cfg", "cfg_large", "cfg_opt", "spatial_cfg_opt", "Raw_SD", "Raw_TD", "SD_TD", "TD_GD"], default="original", help="Computation strategy for analysis")
    parser.add_argument("--ref_first", action="store_true", default=False, help="Use first row/column as reference for spatial difference")
    parser.add_argument("--bit_th", type=int, default=4, help="Bit threshold for large numbers computation")
    parser.add_argument("--clamp", action="store_true", default=False, help="Clamp input activations into int8")
    parser.add_argument("--analysis", type=str, choices=["bitwidth", "similarity", "shape"], default=None,  help="Choose from: bitwidth, similarity, shape")
    parser.add_argument("--similarity-types", type=str, choices=["spatial", "temporal", "conditional"], default=None, help="Types of similarity analysis to perform")
    parser.add_argument("--sample", action="store_true", default=False, help="generate samples")
    parser.add_argument("--bitslice-method", type=str, choices=["bada", "naive", "sibia", "none"], default="none", help="Bitslice method")
    parser.add_argument("--bitslice-bits", type=int, default=9, help="Bitwidth for bitslice method")
    parser.add_argument("--bitslice-width", type=int, default=3, help="Width of each bitslice")
    parser.add_argument("--bitslice-keep-overflow", action="store_true", default=False, help="Keep overflow slice in bitslice method")
    parser.add_argument("--scheme", type=int, default=2, help="hardware optimizatin scheme")
    parser.add_argument("--path-reverse", action="store_true", default=False, help="(uncond, cond)=(method1, method2) by default, path_reverse to conduct (uncond, cond)=(method2, method1)")
    parser.add_argument("--save-tile", action="store_true", default=False, help="Save tile patterns")
    parser.add_argument("--performance", action="store_true", default=False, help="Performance analysis rather than bitslice analysis")
    ##################################################
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    os.makedirs(opt.outdir, exist_ok=True)
    
    ##################################################
    if opt.sample:
        outpath = os.path.join(opt.outdir, "samples")
    else:
        # outpath = os.path.join(opt.outdir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        outpath = os.path.join(opt.outdir, "Final")
        # outpath = os.path.join(opt.outdir, "test")
    ##################################################

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    log_path = os.path.join(outpath, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Arguments: {opt}")
    logger.info(f"Saving to {outpath}")

    ##################################################
    logger.info(f"===== Arguments =====")
    logger.info(f"--precision: {opt.precision}")
    logger.info(f"--hook: {opt.hook}")
    logger.info(f"--strategy: {opt.strategy}")
    logger.info(f"--ref_first: {opt.ref_first}")
    logger.info(f"--bit_th: {opt.bit_th}")
    logger.info(f"--clamp: {opt.clamp}")
    logger.info(f"--analysis: {opt.analysis}")
    logger.info(f"--similarity-types: {opt.similarity_types}")
    logger.info(f"--bitslice-method: {opt.bitslice_method}")
    logger.info(f"--bitslice-bits: {opt.bitslice_bits}")
    logger.info(f"--bitslice-width: {opt.bitslice_width}")
    logger.info(f"--bitslice-keep-overflow: {opt.bitslice_keep_overflow}")
    logger.info(f"--scheme: {opt.scheme}")
    logger.info(f"--path-reverse: {opt.path_reverse}")
    logger.info(f"--save-tile: {opt.save_tile}")
    logger.info(f"--performance: {opt.performance}")
    logger.info(f"===== Arguments =====\n")
    ##################################################

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    assert(opt.cond)
    if opt.ptq:
        if opt.split:
            setattr(sampler.model.model.diffusion_model, "split", True)
        if opt.quant_mode == 'qdiff':
            wq_params = {'n_bits': opt.weight_bit, 'channel_wise': True, 'scale_method': 'mse'}
            aq_params = {'n_bits': opt.act_bit, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param':  opt.quant_act}
            if opt.resume:
                logger.info('Load with min-max quick initialization')
                wq_params['scale_method'] = 'max'
                aq_params['scale_method'] = 'max'
            if opt.resume_w:
                wq_params['scale_method'] = 'max'
            qnn = QuantModel(
                model=sampler.model.model.diffusion_model, weight_quant_params=wq_params, act_quant_params=aq_params,
                act_quant_mode="qdiff", sm_abit=opt.sm_abit)
            qnn.cuda()
            qnn.eval()
            logging.info(qnn)

            if opt.no_grad_ckpt:
                logger.info('Not use gradient checkpointing for transformer blocks')
                qnn.set_grad_ckpt(False)

            if opt.resume:
                cali_data = (torch.randn(1, 4, 64, 64), torch.randint(0, 1000, (1,)), torch.randn(1, 77, 768))
                resume_cali_model(qnn, opt.cali_ckpt, cali_data, opt.quant_act, "qdiff", cond=opt.cond)
            else:
                logger.info(f"Sampling data from {opt.cali_st} timesteps for calibration")
                sample_data = torch.load(opt.cali_data_path)
                cali_data = get_train_samples(opt, sample_data, opt.ddim_steps)
                del(sample_data)
                gc.collect()
                logger.info(f"Calibration data shape: {cali_data[0].shape} {cali_data[1].shape} {cali_data[2].shape}")

                cali_xs, cali_ts, cali_cs = cali_data
                if opt.resume_w:
                    resume_cali_model(qnn, opt.cali_ckpt, cali_data, False, cond=opt.cond)
                else:
                    logger.info("Initializing weight quantization parameters")
                    qnn.set_quant_state(True, False) # enable weight quantization, disable act quantization
                    _ = qnn(cali_xs[:8].cuda(), cali_ts[:8].cuda(), cali_cs[:8].cuda())
                    logger.info("Initializing has done!") 
                # Kwargs for weight rounding calibration
                kwargs = dict(cali_data=cali_data, batch_size=opt.cali_batch_size, 
                            iters=opt.cali_iters, weight=0.01, asym=True, b_range=(20, 2),
                            warmup=0.2, act_quant=False, opt_mode='mse', cond=opt.cond)
                
                def recon_model(model):
                    """
                    Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
                    """
                    for name, module in model.named_children():
                        logger.info(f"{name} {isinstance(module, BaseQuantBlock)}")
                        if name == 'output_blocks':
                            logger.info("Finished calibrating input and mid blocks, saving temporary checkpoint...")
                            in_recon_done = True
                            torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
                        if name.isdigit() and int(name) >= 9:
                            logger.info(f"Saving temporary checkpoint at {name}...")
                            torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))
                            
                        if isinstance(module, QuantModule):
                            if module.ignore_reconstruction is True:
                                logger.info('Ignore reconstruction of layer {}'.format(name))
                                continue
                            else:
                                logger.info('Reconstruction for layer {}'.format(name))
                                layer_reconstruction(qnn, module, **kwargs)
                        elif isinstance(module, BaseQuantBlock):
                            if module.ignore_reconstruction is True:
                                logger.info('Ignore reconstruction of block {}'.format(name))
                                continue
                            else:
                                logger.info('Reconstruction for block {}'.format(name))
                                block_reconstruction(qnn, module, **kwargs)
                        else:
                            recon_model(module)

                if not opt.resume_w:
                    logger.info("Doing weight calibration")
                    recon_model(qnn)
                    qnn.set_quant_state(weight_quant=True, act_quant=False)
                if opt.quant_act:
                    logger.info("UNet model")
                    logger.info(model.model)                    
                    logger.info("Doing activation calibration")
                    # Initialize activation quantization parameters
                    qnn.set_quant_state(True, True)
                    with torch.no_grad():
                        inds = np.random.choice(cali_xs.shape[0], 16, replace=False)
                        _ = qnn(cali_xs[inds].cuda(), cali_ts[inds].cuda(), cali_cs[inds].cuda())
                        if opt.running_stat:
                            logger.info('Running stat for activation quantization')
                            inds = np.arange(cali_xs.shape[0])
                            np.random.shuffle(inds)
                            qnn.set_running_stat(True, opt.rs_sm_only)
                            for i in trange(int(cali_xs.size(0) / 16)):
                                _ = qnn(cali_xs[inds[i * 16:(i + 1) * 16]].cuda(), 
                                    cali_ts[inds[i * 16:(i + 1) * 16]].cuda(),
                                    cali_cs[inds[i * 16:(i + 1) * 16]].cuda())
                            qnn.set_running_stat(False, opt.rs_sm_only)

                    kwargs = dict(
                        cali_data=cali_data, batch_size=opt.cali_batch_size, iters=opt.cali_iters_a, act_quant=True, 
                        opt_mode='mse', lr=opt.cali_lr, p=opt.cali_p, cond=opt.cond)
                    recon_model(qnn)
                    qnn.set_quant_state(weight_quant=True, act_quant=True)
                
                logger.info("Saving calibrated quantized UNet model")
                for m in qnn.model.modules():
                    if isinstance(m, AdaRoundQuantizer):
                        m.zero_point = nn.Parameter(m.zero_point)
                        m.delta = nn.Parameter(m.delta)
                    elif isinstance(m, UniformAffineQuantizer) and opt.quant_act:
                        if m.zero_point is not None:
                            if not torch.is_tensor(m.zero_point):
                                m.zero_point = nn.Parameter(torch.tensor(float(m.zero_point)))
                            else:
                                m.zero_point = nn.Parameter(m.zero_point)
                torch.save(qnn.state_dict(), os.path.join(outpath, "ckpt.pth"))

            sampler.model.model.diffusion_model = qnn

    logging.info("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        logging.info(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    # write config out
    sampling_file = os.path.join(outpath, "sampling_config.yaml")
    sampling_conf = vars(opt)
    with open(sampling_file, 'a+') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    if opt.verbose:
        logger.info("UNet model")
        logger.info(model.model)

    ##############################################
    ###  Hooks for difference computing        ###
    ##############################################
    if opt.hook:
        from analyses.manager import TensorAnalysisManager
        from hooks.utils import register_analysis_hooks, remove_hooks

        analyzer_configs = {}
        if opt.analysis == "similarity" and opt.similarity_types:
            analyzer_configs['similarity'] = {'active_analysis': opt.similarity_types, 'ref_first': opt.ref_first}
        output_dir = outpath if opt.bitslice_method != "none" else None    #! only for bitslice performance
        analysis_manager = TensorAnalysisManager(active_analyzer=opt.analysis, analyzer_configs=analyzer_configs, output_dir=output_dir)

        # include_modules = []
        # exclude_modules = []

        # total 258 modules hooked
        include_modules = None
        exclude_modules = ["time_embed.0", "time_embed.2", "emb_layers.1"]
        
        # include_modules = ["input_blocks.1.1.proj_in"]
        # exclude_modules = None

        # total 1 module hooked (save tile)
        # include_modules = ["input_blocks.1.1.transformer_blocks.0.ff.net.2"]
        # include_modules = ["output_blocks.11.1.transformer_blocks.0.ff.net.2"]
        # include_modules = ["input_blocks.1.1.transformer_blocks.0.ff.net.2", "output_blocks.11.1.transformer_blocks.0.ff.net.2"]
        # exclude_modules = None

        logger.info(f"include_modules: {include_modules}")
        logger.info(f"exclude_modules: {exclude_modules}")

        strategy_kwargs = {
            "ref_first": opt.ref_first,
            "bit_th"   : opt.bit_th,
            "clamp"    : opt.clamp,
            # "save"     : True,  #!
            # "save"     : False,
            "save"     : opt.save_tile,
            "scheme"   : opt.scheme,
            "path_reverse": opt.path_reverse,
            "performance": opt.performance,
        }
        
        bitslice_kwargs = {
            "bits"            : opt.bitslice_bits,
            "slice_width"     : opt.bitslice_width,
            "discard_overflow": not opt.bitslice_keep_overflow
        }

        saver_kwargs = {
            # "enabled": True,    #!
            # "enabled": False,
            "enabled": opt.save_tile,
            "save_dir": outpath,
        }

        handles = register_analysis_hooks(
            qnn.model, 
            analysis_manager,
            computation_strategy=opt.strategy,
            bitslice_method=opt.bitslice_method,
            bitslice_kwargs=bitslice_kwargs,
            include_modules=include_modules,
            exclude_modules=exclude_modules,
            saver_kwargs=saver_kwargs,
            **strategy_kwargs
        )

    ### simple hook
    # from functools import partial
    # from hook import hook_original
    # exclude_modules = ["diffusion_model.model.time_embed.0", "diffusion_model.model.time_embed.2"]
    # print("exclude_modules:", exclude_modules)

    # for name, module in model.model.named_modules():
    #     if isinstance(module, QuantModule):
    #         if name not in exclude_modules:
    #             logger.info(f"{name}, fwd_func:{module.fwd_func}")
    #             module.register_forward_hook(partial(hook_original, name=name))

    # model.model.diffusion_model.model.input_blocks[0][0].register_forward_hook(partial(hook_original, name="model.model.diffusion_model.model.input_blocks.0.0"))

    ##############################################

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                # for n in trange(opt.n_iter, desc="Sampling"):
                #     for prompts in tqdm(data, desc="data"):
                for p_idx, prompts in enumerate(tqdm(data, desc="data")):
                    for n in trange(opt.n_iter, desc="Sampling"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=opt.n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=opt.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=opt.ddim_eta,
                                                         x_T=start_code)

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image = x_samples_ddim
                        # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        if not opt.skip_save:
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_checked_image_torch)

                        ##################################################
                        if opt.hook:
                            # Save analysis results for this round
                            if handles and analysis_manager.analyzers:
                                analysis_dir = os.path.join(outpath, f"prompt{p_idx}/round{n}")
                                analysis_manager.save_all_results(analysis_dir)
                                logger.info(f"Analysis results saved for prompt{p_idx} round {n}")
                                
                                # Clear results for next round
                                analysis_manager.clear_results()
                        ##################################################

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                    grid_count += 1

                toc = time.time()

                ##################################################
                if opt.hook:
                    if handles:
                        remove_hooks(handles)
                ##################################################

    logging.info(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
