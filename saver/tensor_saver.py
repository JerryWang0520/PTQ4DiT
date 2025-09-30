import os
import numpy as np
import torch


class TensorSaver:
    def __init__(self, enabled=False, save_dir=None, global_tile_configs=None, local_tile_configs=None):
        self.enabled = enabled
        self.save_dir = save_dir
        self.global_tile_m00_k00_n00 = {}

        self.default_global_tile_configs = {
            'input' : (128, 256),
            'weight': (256, 256), 
            'output': (128, 256)
        }

        self.default_local_tile_configs = {
            'input' : (8 , 32),
            'weight': (32, 16),
            'output': (8 , 16)
        }

        self.global_tile_configs = global_tile_configs or self.default_global_tile_configs
        self.local_tile_configs  = local_tile_configs  or self.default_local_tile_configs

    def _get_global_tile_configs(self, tensor_type):
        return self.global_tile_configs.get(tensor_type, self.default_global_tile_configs.get(tensor_type))

    def _get_local_tile_configs(self, tensor_type):
        return self.local_tile_configs.get(tensor_type, self.default_local_tile_configs.get(tensor_type))
    
    def _generate_tiles(self, matrix, tile_size):
        rows, cols = matrix.shape
        tile_h, tile_w = tile_size
        
        tiles = []
        for r_idx in range(0, rows, tile_h):
            for c_idx in range(0, cols, tile_w):
                r_end = min(r_idx + tile_h, rows)
                c_end = min(c_idx + tile_w, cols)
                
                tile = matrix[r_idx:r_end, c_idx:c_end]
                
                if tile.shape != tile_size:
                    pad_h = tile_h - tile.shape[0]
                    pad_w = tile_w - tile.shape[1]
                    tile = np.pad(tile, ((0, pad_h), (0, pad_w)))
                
                tiles.append({
                    'tile': tile,
                    'indices': (r_idx // tile_h, c_idx // tile_w),
                    # 'coords': (r_idx, c_idx)
                })
        
        return tiles
    
    def save_global_tiles(self, tensors_dict, name, step):
        if not self.enabled or not (step == 2 or step == 50):
            return
        
        save_dir = self.save_dir or "./output/patterns"
        dir_name = self._get_dir_name(name, step)        
        base_dir = os.path.join(save_dir, "active", dir_name)
        os.makedirs(base_dir, exist_ok=True)

        # Save zero points (unchanged)
        self._save_zero_points(tensors_dict, base_dir)
        
        # Save input tensors with configurable tiling
        if "x_q" in tensors_dict:
            self._save_input_tensors(tensors_dict["x_q"], base_dir, "act", signed=False, bits=8)
            
        if "x_raw" in tensors_dict:
            self._save_input_tensors(tensors_dict["x_raw"], base_dir, "x_raw", signed=True, bits=9)
            
        # Save weight tensors with configurable tiling  
        if "w_q" in tensors_dict:
            self._save_weight_tensors(tensors_dict["w_q"], base_dir, "wgt", signed=False, bits=8)
            
        if "w_raw" in tensors_dict:
            self._save_weight_tensors(tensors_dict["w_raw"], base_dir, "w_raw", signed=True, bits=9)
            
        # Save output tensors with configurable tiling
        if "xw_raw" in tensors_dict:
            self._save_output_tensors(tensors_dict["xw_raw"], base_dir, "out", signed=True, bits=24)
        
        print(f"Saved tensors in memory layout to {base_dir}")

    def _save_input_tensors(self, tensor, base_dir, prefix, signed=False, bits=8):
        tile_size = self._get_global_tile_configs('input')
        
        tensor0 = tensor[0].cpu().numpy()
        tensor1 = tensor[1].cpu().numpy()
        
        tiles0 = self._generate_tiles(tensor0, tile_size)
        tiles1 = self._generate_tiles(tensor1, tile_size)
        
        for tile_info0, tile_info1 in zip(tiles0, tiles1):
            m_idx, k_idx = tile_info0['indices']
            tile_dir = os.path.join(base_dir, f"{prefix}_m{m_idx:02d}_k{k_idx:02d}")
            os.makedirs(tile_dir, exist_ok=True)
            
            self._save_input_matrix_global(tile_info0['tile'], tile_dir, "gibuf0", signed=signed, bits=bits)
            self._save_input_matrix_global(tile_info1['tile'], tile_dir, "gibuf1", signed=signed, bits=bits)
            
            # Store first tile for local processing
            if m_idx == 0 and k_idx == 0 and prefix == "x_raw":
                self.global_tile_m00_k00_n00["input0"] = tile_info0['tile']
                self.global_tile_m00_k00_n00["input1"] = tile_info1['tile']

    def _save_weight_tensors(self, tensor, base_dir, prefix, signed=False, bits=8):
        """Save weight tensors with configurable tiling"""
        tile_size = self._get_global_tile_configs('weight')
        tensor = tensor.cpu().numpy()
        
        tiles = self._generate_tiles(tensor, tile_size)
        
        for tile_info in tiles:
            k_idx, n_idx = tile_info['indices']
            tile_dir = os.path.join(base_dir, f"{prefix}_k{k_idx:02d}_n{n_idx:02d}")
            os.makedirs(tile_dir, exist_ok=True)
            
            self._save_weight_matrix_global(tile_info['tile'], tile_dir, "gwbuf", signed=signed, bits=bits)
            
            # Store first tile for local processing
            if k_idx == 0 and n_idx == 0 and prefix == "w_raw":
                self.global_tile_m00_k00_n00["weight"] = tile_info['tile']

    def _save_output_tensors(self, tensor, base_dir, prefix, signed=True, bits=24):
        tile_size = self._get_global_tile_configs('output')
        
        tensor0 = tensor[0].cpu().numpy()
        tensor1 = tensor[1].cpu().numpy()
        
        tiles0 = self._generate_tiles(tensor0, tile_size)
        tiles1 = self._generate_tiles(tensor1, tile_size)
        
        for tile_info0, tile_info1 in zip(tiles0, tiles1):
            m_idx, n_idx = tile_info0['indices']
            tile_dir = os.path.join(base_dir, f"{prefix}_m{m_idx:02d}_n{n_idx:02d}")
            os.makedirs(tile_dir, exist_ok=True)
            
            self._save_output_matrix_global(tile_info0['tile'], tile_dir, "gobuf", bank_offset=0, signed=signed, bits=bits)
            self._save_output_matrix_global(tile_info1['tile'], tile_dir, "gobuf", bank_offset=4, signed=signed, bits=bits)

    def _save_zero_points(self, tensors_dict, base_dir):
        if "x_zp" in tensors_dict:
            zp_dir = os.path.join(base_dir, "zero_points")
            os.makedirs(zp_dir, exist_ok=True)
            x_zp = tensors_dict["x_zp"].cpu().numpy().flatten()
            with open(os.path.join(zp_dir, "x_zp.txt"), "w") as f:
                for val in x_zp:
                    f.write(self.uint_to_bin(val, 8) + "\n")
        
        if "w_zp" in tensors_dict:
            zp_dir = os.path.join(base_dir, "zero_points")
            os.makedirs(zp_dir, exist_ok=True)
            w_zp = tensors_dict["w_zp"].cpu().numpy().flatten()

            tile_size = self._get_global_tile_configs('weight')
            tile_h, tile_w = tile_size

            if len(w_zp) % tile_w != 0:
                pad_size = tile_w - len(w_zp) % tile_w
                w_zp = np.pad(w_zp, (0, pad_size))

            with open(os.path.join(zp_dir, "w_zp.txt"), "w") as f:
                for val in w_zp:
                    f.write(self.uint_to_bin(val, 8) + "\n")
    

    def save_local_tiles(self, name, step):
        if not self.enabled or not (step == 2 or step == 50):
            return
        
        if not hasattr(self, 'global_tile_m00_k00_n00') or not self.global_tile_m00_k00_n00:
            print("No global tiles saved for local extraction")
            return
        
        save_dir = self.save_dir or "./output/patterns"        
        dir_name = self._get_dir_name(name, step)
        base_dir = os.path.join(save_dir, "active", dir_name)
        
        input0 = self.global_tile_m00_k00_n00.get("input0")
        input1 = self.global_tile_m00_k00_n00.get("input1") 
        weight = self.global_tile_m00_k00_n00.get("weight")
        
        if input0 is None or input1 is None or weight is None:
            print("Missing required global tiles for local extraction")
            return
        
        input_tile_size         = self._get_local_tile_configs('input')
        weight_tile_size        = self._get_local_tile_configs('weight')
        output_tile_size        = self._get_local_tile_configs('output')
        global_output_tile_size = self._get_global_tile_configs('output')
        
        #* Save local input tiles
        input0_tiles = self._generate_tiles(input0, input_tile_size)
        input1_tiles = self._generate_tiles(input1, input_tile_size)
        
        for tile_info0, tile_info1 in zip(input0_tiles, input1_tiles):
            m_idx, k_idx = tile_info0['indices']
            local_dir = os.path.join(base_dir, "x_raw_m00_k00", f"local_m{m_idx:02d}_k{k_idx:02d}")
            self._save_input_matrix_local(tile_info0['tile'], local_dir, "libuf0", signed=True, bits=9)
            self._save_input_matrix_local(tile_info1['tile'], local_dir, "libuf1", signed=True, bits=9)
        
        #* Save local weight tiles  
        weight_tiles = self._generate_tiles(weight, weight_tile_size)
        
        for tile_info in weight_tiles:
            k_idx, n_idx = tile_info['indices']
            local_dir = os.path.join(base_dir, "w_raw_k00_n00", f"local_k{k_idx:02d}_n{n_idx:02d}")
            self._save_weight_matrix_local(tile_info['tile'], local_dir, "lwbuf", signed=True, bits=9)
        
        # Compute and save local output tiles
        global_psum0 = np.zeros(global_output_tile_size)
        global_psum1 = np.zeros(global_output_tile_size)
        
        m_tiles = global_output_tile_size[0] // output_tile_size[0]
        n_tiles = global_output_tile_size[1] // output_tile_size[1] 
        k_tiles = input0.shape[1] // input_tile_size[1]
            
        for m_tile_idx in range(m_tiles):
            for n_tile_idx in range(n_tiles):
                output_tile0 = np.zeros(output_tile_size)
                output_tile1 = np.zeros(output_tile_size)
                
                for k_tile_idx in range(k_tiles):
                    input_tile_idx  = m_tile_idx * k_tiles + k_tile_idx
                    weight_tile_idx = k_tile_idx * n_tiles + n_tile_idx
                    
                    if input_tile_idx < len(input0_tiles) and weight_tile_idx < len(weight_tiles):
                        input_tile0 = input0_tiles[input_tile_idx ]['tile']
                        input_tile1 = input1_tiles[input_tile_idx ]['tile'] 
                        weight_tile = weight_tiles[weight_tile_idx]['tile']
                        
                        output_tile0 += np.matmul(input_tile0, weight_tile)
                        output_tile1 += np.matmul(input_tile1, weight_tile)
                
                #* Save local output tiles
                local_dir = os.path.join(base_dir, "out_m00_n00_k00", f"local_m{m_tile_idx:02d}_n{n_tile_idx:02d}")
                self._save_output_matrix_local(output_tile0, local_dir, "lobuf0", signed=True, bits=24)
                self._save_output_matrix_local(output_tile1, local_dir, "lobuf1", signed=True, bits=24)
                
                # Update global output
                m_start = m_tile_idx * output_tile_size[0]
                m_end   = m_start + output_tile_size[0]
                n_start = n_tile_idx * output_tile_size[1]
                n_end   = n_start + output_tile_size[1]
                
                global_psum0[m_start:m_end, n_start:n_end] = output_tile0
                global_psum1[m_start:m_end, n_start:n_end] = output_tile1
        
        #* Save global output
        global_dir = os.path.join(base_dir, "out_m00_n00_k00")
        self._save_output_matrix_global(global_psum0, global_dir, "gobuf", bank_offset=0, signed=True, bits=24)
        self._save_output_matrix_global(global_psum1, global_dir, "gobuf", bank_offset=4, signed=True, bits=24)

        print(f"Saved local tiles to {base_dir}")


    # def save_tensors(self, tensors_dict, name, step):
    #     if not self.enabled:
    #         return
            
    #     save_dir = os.path.join(self.save_dir, "tensor_data", name)
    #     os.makedirs(save_dir, exist_ok=True)
        
    #     filename = f"step_{step}.npz"
    #     filepath = os.path.join(save_dir, filename)
        
    #     save_data = {}
    #     for key, tensor in tensors_dict.items():
    #         save_data[key] = tensor.detach().cpu().numpy()
        
    #     np.savez_compressed(filepath, **save_data)


    # def save_global_tiles(self, tensors_dict, name, step):
    #     if not self.enabled or step != 2 or step != 50:
    #         return
        
    #     save_dir = self.save_dir or "./output/patterns"
    #     dir_name = self._get_dir_name(name, step)        
    #     base_dir = os.path.join(save_dir, "active", dir_name)
    #     os.makedirs(base_dir, exist_ok=True)

    #     if "x_zp" in tensors_dict:
    #         zp_dir = os.path.join(base_dir, "zero_points")
    #         os.makedirs(zp_dir, exist_ok=True)
            
    #         x_zp = tensors_dict["x_zp"].cpu().numpy().flatten()
            
    #         with open(os.path.join(zp_dir, "x_zp.txt"), "w") as f:
    #             for val in x_zp:
    #                 f.write(self.uint_to_bin(val, 8) + "\n")
        
    #     if "w_zp" in tensors_dict:
    #         zp_dir = os.path.join(base_dir, "zero_points")
    #         os.makedirs(zp_dir, exist_ok=True)
            
    #         w_zp = tensors_dict["w_zp"].cpu().numpy().flatten()

    #         if len(w_zp) % 256 != 0:
    #             pad_size = 256 - len(w_zp) % 256
    #             w_zp = np.pad(w_zp, (0, pad_size))

    #         with open(os.path.join(zp_dir, "w_zp.txt"), "w") as f:
    #             for val in w_zp:
    #                 f.write(self.uint_to_bin(val, 8) + "\n")

    #     if "x_q" in tensors_dict:
    #         act_tensor = tensors_dict["x_q"]
    #         batch0 = act_tensor[0].cpu().numpy()
    #         batch1 = act_tensor[1].cpu().numpy()

    #         for m_idx in range(0, 256, 128):
    #             for k_idx in range(0, 4608, 256):
    #                 m_end = min(m_idx + 128, 256)
    #                 k_end = min(k_idx + 256, 4608)

    #                 tile0 = batch0[m_idx:m_end, k_idx:k_end]
    #                 tile1 = batch1[m_idx:m_end, k_idx:k_end]

    #                 if tile0.shape != (128, 256):
    #                     pad_h = 128 - tile0.shape[0]
    #                     pad_w = 256 - tile0.shape[1]
    #                     tile0 = np.pad(tile0, ((0, pad_h), (0, pad_w)))
    #                     tile1 = np.pad(tile1, ((0, pad_h), (0, pad_w)))

    #                 tile_dir = os.path.join(base_dir, f"act_m{m_idx//128:02d}_k{k_idx//256:02d}")
    #                 self._save_input_matrix_global(tile0, tile_dir, "gibuf0", signed=False, bits=8)
    #                 self._save_input_matrix_global(tile1, tile_dir, "gibuf1", signed=False, bits=8)
       
    #     if "w_q" in tensors_dict:
    #         wgt_tensor = tensors_dict["w_q"].cpu().numpy()
            
    #         for k_idx in range(0, 4608, 256):
    #             for n_idx in range(0, 1152, 256):
    #                 k_end = min(k_idx + 256, 4608)
    #                 n_end = min(n_idx + 256, 1152)
                    
    #                 tile = wgt_tensor[k_idx:k_end, n_idx:n_end]
                    
    #                 if tile.shape != (256, 256):
    #                     pad_h = 256 - tile.shape[0]
    #                     pad_w = 256 - tile.shape[1]
    #                     tile = np.pad(tile, ((0, pad_h), (0, pad_w)))
                    
    #                 tile_dir = os.path.join(base_dir, f"wgt_k{k_idx//256:02d}_n{n_idx//256:02d}")
    #                 self._save_weight_matrix_global(tile, tile_dir, "gwbuf", signed=False, bits=8)
        
    #     if "x_raw" in tensors_dict:
    #         act_tensor = tensors_dict["x_raw"]
    #         batch0 = act_tensor[0].cpu().numpy()
    #         batch1 = act_tensor[1].cpu().numpy()

    #         for m_idx in range(0, 256, 128):
    #             for k_idx in range(0, 4608, 256):
    #                 m_end = min(m_idx + 128, 256)
    #                 k_end = min(k_idx + 256, 4608)

    #                 tile0 = batch0[m_idx:m_end, k_idx:k_end]
    #                 tile1 = batch1[m_idx:m_end, k_idx:k_end]

    #                 if tile0.shape != (128, 256):
    #                     pad_h = 128 - tile0.shape[0]
    #                     pad_w = 256 - tile0.shape[1]
    #                     tile0 = np.pad(tile0, ((0, pad_h), (0, pad_w)))
    #                     tile1 = np.pad(tile1, ((0, pad_h), (0, pad_w)))

    #                 tile_dir = os.path.join(base_dir, f"x_raw_m{m_idx//128:02d}_k{k_idx//256:02d}")
    #                 self._save_input_matrix_global(tile0, tile_dir, "gibuf0", signed=True, bits=9)
    #                 self._save_input_matrix_global(tile1, tile_dir, "gibuf1", signed=True, bits=9)
        
    #                 if m_idx == 0 and k_idx == 0:
    #                     self.global_tile_m00_k00_n00["input0"] = tile0
    #                     self.global_tile_m00_k00_n00["input1"] = tile1

    #     if "w_raw" in tensors_dict:
    #         wgt_tensor = tensors_dict["w_raw"].cpu().numpy()
            
    #         for k_idx in range(0, 4608, 256):
    #             for n_idx in range(0, 1152, 256):
    #                 k_end = min(k_idx + 256, 4608)
    #                 n_end = min(n_idx + 256, 1152)
                    
    #                 tile = wgt_tensor[k_idx:k_end, n_idx:n_end]
                    
    #                 if tile.shape != (256, 256):
    #                     pad_h = 256 - tile.shape[0]
    #                     pad_w = 256 - tile.shape[1]
    #                     tile = np.pad(tile, ((0, pad_h), (0, pad_w)))
                    
    #                 tile_dir = os.path.join(base_dir, f"w_raw_k{k_idx//256:02d}_n{n_idx//256:02d}")
    #                 self._save_weight_matrix_global(tile, tile_dir, "gwbuf", signed=True, bits=9)

    #                 if k_idx == 0 and n_idx == 0:
    #                     self.global_tile_m00_k00_n00["weight"] = tile
        
    #     if "xw_raw" in tensors_dict:
    #         out_tensor = tensors_dict["xw_raw"]
    #         batch0 = out_tensor[0].cpu().numpy()
    #         batch1 = out_tensor[1].cpu().numpy()
            
    #         for m_idx in range(0, 256, 128):
    #             for n_idx in range(0, 1152, 256):
    #                 m_end = min(m_idx + 128, 256)
    #                 n_end = min(n_idx + 256, 1152)
                    
    #                 tile0 = batch0[m_idx:m_end, n_idx:n_end]
    #                 tile1 = batch1[m_idx:m_end, n_idx:n_end]
                    
    #                 if tile0.shape != (128, 256):
    #                     pad_h = 128 - tile0.shape[0]
    #                     pad_w = 256 - tile0.shape[1]
    #                     tile0 = np.pad(tile0, ((0, pad_h), (0, pad_w)))
    #                     tile1 = np.pad(tile1, ((0, pad_h), (0, pad_w)))
                    
    #                 tile_dir = os.path.join(base_dir, f"out_m{m_idx//128:02d}_n{n_idx//256:02d}")
    #                 self._save_output_matrix_global(tile0, tile_dir, "gobuf", bank_offset=0, signed=True, bits=24)
    #                 self._save_output_matrix_global(tile1, tile_dir, "gobuf", bank_offset=4, signed=True, bits=24)
        
    #     print(f"Saved tensors in memory layout to {base_dir}")

    # def save_local_tiles(self, name, step):
    #     if not self.enabled or step != 2 or step != 50:
    #         return
        
    #     if not hasattr(self, 'global_tile_m00_k00_n00') or not self.global_tile_m00_k00_n00:
    #         print("No global tiles saved for local extraction")
    #         return
        
    #     save_dir = self.save_dir or "./output/patterns"        
    #     dir_name = self._get_dir_name(name, step)
    #     base_dir = os.path.join(save_dir, "active", dir_name)
        
    #     input0 = self.global_tile_m00_k00_n00.get("input0")
    #     input1 = self.global_tile_m00_k00_n00.get("input1") 
    #     weight = self.global_tile_m00_k00_n00.get("weight")
        
    #     if input0 is None or input1 is None or weight is None:
    #         print("Missing required global tiles for local extraction")
    #         return
        
    #     for m_idx in range(16):
    #         for k_idx in range(8):
    #             input_tile0 = self._extract_tile(input0, m_idx, k_idx, tile_size=(8, 32))
    #             input_tile1 = self._extract_tile(input1, m_idx, k_idx, tile_size=(8, 32))
                
    #             local_dir = os.path.join(base_dir, "x_raw_m00_k00", f"local_m{m_idx:02d}_k{k_idx:02d}")
    #             self._save_input_matrix_local(input_tile0, local_dir, "libuf0", signed=True, bits=9)
    #             self._save_input_matrix_local(input_tile1, local_dir, "libuf1", signed=True, bits=9)
        
    #     for k_idx in range(8):
    #         for n_idx in range(16):
    #             weight_tile = self._extract_tile(weight, k_idx, n_idx, tile_size=(32, 16))
                
    #             local_dir = os.path.join(base_dir, "w_raw_k00_n00", f"local_k{k_idx:02d}_n{n_idx:02d}")
    #             self._save_weight_matrix_local(weight_tile, local_dir, "lwbuf", signed=True, bits=9)
        
    #     global_psum0 = np.zeros((128, 256))
    #     global_psum1 = np.zeros((128, 256))

    #     for m_idx in range(16):
    #         for n_idx in range(16):
    #             output_tile0 = np.zeros((8, 16))
    #             output_tile1 = np.zeros((8, 16))
                
    #             for k_idx in range(8):
    #                 input_tile0 = self._extract_tile(input0, m_idx, k_idx, tile_size=(8, 32))
    #                 input_tile1 = self._extract_tile(input1, m_idx, k_idx, tile_size=(8, 32))
    #                 weight_tile = self._extract_tile(weight, k_idx, n_idx, tile_size=(32, 16))
                    
    #                 output_tile0 += np.matmul(input_tile0, weight_tile)
    #                 output_tile1 += np.matmul(input_tile1, weight_tile)
                
    #             m_start, m_end = m_idx * 8, (m_idx + 1) * 8
    #             n_start, n_end = n_idx * 16, (n_idx + 1) * 16
    #             global_psum0[m_start:m_end, n_start:n_end] = output_tile0
    #             global_psum1[m_start:m_end, n_start:n_end] = output_tile1

    #             local_dir = os.path.join(base_dir, "out_m00_n00_k00", f"local_m{m_idx:02d}_n{n_idx:02d}")
    #             self._save_output_matrix_local(output_tile0, local_dir, "lobuf0", signed=True, bits=24)
    #             self._save_output_matrix_local(output_tile1, local_dir, "lobuf1", signed=True, bits=24)
        
    #     global_dir = os.path.join(base_dir, "out_m00_n00_k00")
    #     self._save_output_matrix_global(global_psum0, global_dir, "gobuf", bank_offset=0, signed=True, bits=24)
    #     self._save_output_matrix_global(global_psum1, global_dir, "gobuf", bank_offset=4, signed=True, bits=24)

    #     print(f"Saved local tiles to {base_dir}")

    # def _extract_tile(self, matrix, tile_row_idx, tile_col_idx, tile_size):
    #     tile_height, tile_width = tile_size
    #     tile_row_start = tile_row_idx * tile_height
    #     tile_row_end   = tile_row_start + tile_height
    #     tile_col_start = tile_col_idx * tile_width  
    #     tile_col_end   = tile_col_start + tile_width
        
    #     return matrix[tile_row_start:tile_row_end, tile_col_start:tile_col_end]

    def int_to_bin(self, val, bits):
        min_val = -(1 << (bits - 1))
        max_val =  (1 << (bits - 1)) - 1
        val = int(np.clip(val, min_val, max_val))
        if val < 0:
            val = (1 << bits) + val
        return format(val, f'0{bits}b')
    
    def uint_to_bin(self, val, bits):
        min_val = 0
        max_val = (1 << bits) - 1
        val = int(np.clip(val, min_val, max_val))
        return format(val, f'0{bits}b')

    def _save_input_matrix_global(self, matrix, save_dir, filename, signed=False, bits=8):
        os.makedirs(save_dir, exist_ok=True)
        rows, cols = matrix.shape
        assert rows == 128 and cols == 256, f"Expected 128x256, got {rows}x{cols}"

        filepath = os.path.join(save_dir, f"{filename}.txt")
        with open(filepath, "w") as f:
            for r in range(128):
                for addr_offset in range(16):
                    row_data = []
                    for group in range(7, -1, -1):
                        group_base = group * 32
                        col_even = group_base + (addr_offset * 2)
                        col_odd  = group_base + (addr_offset * 2) + 1

                        if signed:
                            row_data.append(self.int_to_bin(matrix[r][col_odd ], bits))
                            row_data.append(self.int_to_bin(matrix[r][col_even], bits))
                        else:
                            row_data.append(self.uint_to_bin(matrix[r][col_odd ], bits))
                            row_data.append(self.uint_to_bin(matrix[r][col_even], bits))
                    f.write("_".join(row_data) + "\n")

    def _save_weight_matrix_global(self, matrix, save_dir, filename, signed=False, bits=8):
        os.makedirs(save_dir, exist_ok=True)
        rows, cols = matrix.shape
        assert rows == 256 and cols == 256, f"Expected 256x256, got {rows}x{cols}"
        
        for bank_id in range(8):
            filepath = os.path.join(save_dir, f"{filename}{bank_id}.txt")

            with open(filepath, "w") as f:
                start_row = bank_id * 32
                end_row   = start_row + 32
                
                for r in range(start_row, end_row):
                    for addr_offset in range(16):
                        row_data = []
                        start_col = addr_offset * 16
                        for c in range(start_col + 15, start_col - 1, -1):
                            if signed:
                                row_data.append(self.int_to_bin(matrix[r][c], bits))
                            else:
                                row_data.append(self.uint_to_bin(matrix[r][c], bits))
                        f.write("_".join(row_data) + "\n")

    def _save_output_matrix_global(self, matrix, save_dir, filename, bank_offset=0, signed=True, bits=24):
        os.makedirs(save_dir, exist_ok=True)
        rows, cols = matrix.shape
        assert rows == 128 and cols == 256, f"Expected 128x256, got {rows}x{cols}"
       
        for bank_id in range(4):
            filepath = os.path.join(save_dir, f"{filename}{bank_id + bank_offset}.txt")

            with open(filepath, "w") as f:
                for group in range(16):
                    group_base = group * 16
                    bank_base  = group_base + bank_id * 4
                    
                    for r in range(128):
                        row_data = []
                        for c in range(bank_base + 3, bank_base - 1, -1):
                            if signed:
                                row_data.append(self.int_to_bin(matrix[r][c], bits))
                            else:
                                row_data.append(self.uint_to_bin(matrix[r][c], bits))
                        f.write("_".join(row_data) + "\n")

    def _save_input_matrix_local(self, matrix, save_dir, filename, signed=True, bits=9):
        os.makedirs(save_dir, exist_ok=True)
        rows, cols = matrix.shape
        assert rows == 8 and cols == 32, f"Expected 8x32, got {rows}x{cols}"

        filepath = os.path.join(save_dir, f"{filename}.txt")
        with open(filepath, "w") as f:
            for r in range(rows):
                for c in range(cols):
                    if signed:
                        f.write(self.int_to_bin(matrix[r][c], bits) + "\n")
                    else:
                        f.write(self.uint_to_bin(matrix[r][c], bits) + "\n")

    def _save_weight_matrix_local(self, matrix, save_dir, filename, signed=True, bits=9):
        os.makedirs(save_dir, exist_ok=True)
        rows, cols = matrix.shape
        assert rows == 32 and cols == 16, f"Expected 32x16, got {rows}x{cols}"

        filepath = os.path.join(save_dir, f"{filename}.txt")
        with open(filepath, "w") as f:
            for r in range(rows):
                row_data = []
                for c in range(cols - 1, -1, -1):
                    if signed:
                        row_data.append(self.int_to_bin(matrix[r][c], bits))
                    else:
                        row_data.append(self.uint_to_bin(matrix[r][c], bits))
                f.write("_".join(row_data) + "\n")

    def _save_output_matrix_local(self, matrix, save_dir, filename, signed=True, bits=9):
        os.makedirs(save_dir, exist_ok=True)
        rows, cols = matrix.shape
        assert rows == 8 and cols == 16, f"Expected 8x16, got {rows}x{cols}"

        filepath = os.path.join(save_dir, f"{filename}.txt")
        with open(filepath, "w") as f:
            for r in range(rows):
                row_data = []
                for c in range(cols - 1, -1, -1):
                    if signed:
                        row_data.append(self.int_to_bin(matrix[r][c], bits))
                    else:
                        row_data.append(self.uint_to_bin(matrix[r][c], bits))
                f.write("_".join(row_data) + "\n")

    def _get_dir_name(self, name, step):
                
        # StableDiffusion naming: input_blocks.1.1.transformer_blocks.0.ff.net.2
        if any(x in name for x in ["input_blocks", "middle_block", "output_blocks"]):
            parts = name.split(".")
            
            if "input_blocks" in name:
                block_type = "ib"
                block_num = int(parts[1])
            elif "middle_block" in name:
                block_type = "mb"
                block_num = 0
            elif "output_blocks" in name:
                block_type = "ob"
                block_num = int(parts[1])
            
            layer_parts = []
            i = 2
            while i < len(parts):
                if parts[i] == "transformer_blocks":
                    layer_parts.append(f"tb{parts[i+1]}")
                    i += 2
                elif parts[i] == "ff":
                    layer_parts.append("ff")
                    i += 1
                elif parts[i] == "net":
                    layer_parts.append(f"n{parts[i+1]}")
                    i += 2
                elif parts[i] == "attn":
                    layer_parts.append("attn")
                    i += 1
                elif parts[i] in ["to_q", "to_k", "to_v", "to_out"]:
                    layer_parts.append(parts[i])
                    if i+1 < len(parts) and parts[i+1].isdigit():
                        layer_parts[-1] += parts[i+1]
                        i += 2
                    else:
                        i += 1
                else:
                    layer_parts.append(parts[i])
                    i += 1
            
            layer_name = "_".join(layer_parts)
            return f"t{step:02d}_{block_type}{block_num:02d}_{layer_name}"
        
        # DiT naming: blocks.0.mlp.fc1
        elif "blocks." in name:
            block_num = int(name.split("blocks.")[1].split(".")[0])
            layer_name = name.split("blocks.")[1].split(".", 1)[1].replace('.', '_')
            return f"t{step:02d}_b{block_num:02d}_{layer_name}"
        
        else:
            return f"t{step:02d}_{name.replace('.', '_')}"