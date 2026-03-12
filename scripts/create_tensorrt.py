import os
import sys
import logging
import argparse
import numpy as np
import tensorrt as trt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_ttl_config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("EngineBuilder")

class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    Optimized for maximum resource usage.
    """
    def __init__(self, verbose=False, workspace=16, cfg=None):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in GB. Increased for maximum optimization.
        :param cfg: Optional dict from load_ttl_config() for profile shape heuristics.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        
        # Increase the workspace memory pool size for maximum performance
        self.config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace * (2**30)
        )

        self.batch_size = None
        self.network = None
        self.parser = None
        self.cfg = cfg

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        self.network = self.builder.create_network(1)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info(
                "Input '{}' with shape {} and dtype {}".format(
                    input.name, input.shape, input.dtype
                )
            )
        for output in outputs:
            log.info(
                "Output '{}' with shape {} and dtype {}".format(
                    output.name, output.shape, output.dtype
                )
            )

        # assert self.batch_size > 0

    def set_profile(self):
        profile = self.builder.create_optimization_profile()

        # Config-derived dimensions for better heuristics
        se_n_style = self.cfg["se_n_style"] if self.cfg else 50
        ccf = self.cfg["chunk_compress_factor"] if self.cfg else 6

        for i in range(self.network.num_inputs):
            input_tensor = self.network.get_input(i)
            name = input_tensor.name
            shape = input_tensor.shape
            min_shape = []
            opt_shape = []
            max_shape = []
            
            for dim_idx, dim in enumerate(shape):
                if dim == -1:
                    # Dynamic dimension
                    min_shape.append(1)
                    if dim_idx == 0: # Batch size
                        opt_shape.append(1)
                        max_shape.append(8)
                    else: # Other dynamic dims (sequence length etc)
                        opt_shape.append(256)
                        max_shape.append(2048)
                        
                        # ---- Heuristics matching our ONNX exports ----
                        # Text-related axes
                        if "text" in name: 
                            if dim_idx == 1 and len(shape) == 2: # text_ids [B, T]
                                opt_shape[-1] = 128
                                max_shape[-1] = 512
                            elif dim_idx == 2: # text_mask/text_emb last dim
                                opt_shape[-1] = 128
                                max_shape[-1] = 512
                        
                        # Reference / style axes
                        elif "ref" in name or "style" in name:
                            if "z_ref" in name or "ref_mask" in name or "mask" in name:
                                opt_shape[-1] = 256
                                max_shape[-1] = 1024
                            else:
                                # style_ttl T_ref dim (tokens from ref encoder)
                                opt_shape[-1] = se_n_style
                                max_shape[-1] = se_n_style * 4

                        # Latent axes
                        elif "latent" in name or "denoised" in name:
                             if dim_idx == 2:
                                opt_shape[-1] = 512
                                max_shape[-1] = 2048
                        
                        # Vocoder waveform output (PixelShuffle expands T by 512x)
                        elif "waveform" in name:
                            if dim_idx == 2:
                                opt_shape[-1] = 256 * 512
                                max_shape[-1] = 2048 * 512
                else:
                    # Fixed dimension
                    min_shape.append(dim)
                    opt_shape.append(dim)
                    max_shape.append(dim)
            
            log.info(f"Setting profile for {name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
            profile.set_shape(name, min_shape, opt_shape, max_shape)
        
        self.config.add_optimization_profile(profile)

    def create_engine(self, engine_path, precision="fp16", use_int8=False):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16'.
        :param use_int8: Enable INT8 precision mode if hardware supports it.
        """
        engine_path = os.path.realpath(engine_path)
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        # Set precision flags
        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            self.config.set_flag(trt.BuilderFlag.FP16)
        
        if use_int8:
            if not self.builder.platform_has_fast_int8:
                log.warning("INT8 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.INT8)

        # Set optimization profile for dynamic shapes
        self.set_profile()

        # Build and serialize the engine
        engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        if engine_bytes is None:
            log.error("Failed to create engine")
            sys.exit(1)

        with open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine_bytes)

def main(args, cfg=None):
    builder = EngineBuilder(args.verbose, args.workspace, cfg=cfg)
    builder.create_network(args.onnx)
    builder.create_engine(args.engine, args.precision, use_int8=args.use_int8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Support both single file mode (old) and directory mode (new, from README)
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    
    parser.add_argument("--onnx_dir", help="Directory containing ONNX models (batch mode)")
    parser.add_argument("--engine_dir", help="Directory to save TRT engines (batch mode)")

    parser.add_argument(
        "-p", "--precision", default="fp16", choices=["fp32", "fp16"], help="The precision mode to build in, either fp32/fp16"
    )
    parser.add_argument(
        "--use_int8", action="store_true", help="Enable INT8 precision mode (only if supported by the hardware)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    parser.add_argument(
        "-w", "--workspace", default=24, type=int, help="The max memory workspace size to allow in GB"
    )
    parser.add_argument("--config", type=str, default="tts.json", help="Path to tts.json config")
    args = parser.parse_args()

    # Load config
    cfg = None
    if os.path.exists(args.config):
        cfg = load_ttl_config(args.config)
        print(f"[INFO] Loaded config: {args.config} (v{cfg['full_config'].get('tts_version', '?')})")
    else:
        print(f"[WARN] Config {args.config} not found, using default profile heuristics.")

    # Batch Mode
    if args.onnx_dir and args.engine_dir:
        import glob
        os.makedirs(args.engine_dir, exist_ok=True)
        # Only look for regular .onnx files
        onnx_files = glob.glob(os.path.join(args.onnx_dir, "*.onnx"))
        # Filter out any .slim.onnx if they still exist
        onnx_files = [f for f in onnx_files if not f.endswith(".slim.onnx")]
        
        if not onnx_files:
            print(f"No .onnx files found in {args.onnx_dir}")
            sys.exit(1)

        print(f"Found {len(onnx_files)} models to convert: {onnx_files}")
        
        for onnx_path in onnx_files:
            model_name = os.path.basename(onnx_path).replace(".onnx", ".trt")
            
            engine_path = os.path.join(args.engine_dir, model_name)
            
            print(f"--- Converting {onnx_path} -> {engine_path} ---")
            
            # Re-init builder for each model to clear state
            builder = EngineBuilder(args.verbose, args.workspace, cfg=cfg)
            builder.create_network(onnx_path)
            builder.create_engine(engine_path, args.precision, use_int8=args.use_int8)
            
    # Single File Mode
    elif args.onnx and args.engine:
        main(args, cfg=cfg)
    else:
        parser.print_help()
        sys.exit(1)

