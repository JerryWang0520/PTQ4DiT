import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any


class BitsliceStrategy(ABC):
    @abstractmethod
    def reconstruct(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reconstruct tensor using bitslice method"""
        pass
    
    @abstractmethod
    def get_bitslice_info(self, value: int) -> Dict:
        """Get bitslice info for analysis"""
        pass


class BADAStrategy(BitsliceStrategy):
    def __init__(self, bits=9, slice_width=3, discard_overflow=False, bitslice_clamp=False):
        self.bits = bits
        self.slice_width = slice_width
        self.discard_overflow = discard_overflow
        self.bitslice_clamp = bitslice_clamp
        self.device = None

        self.bitslice_map = {}
        self._create_bitslice_map()

        self.gpu_table = None
        self._create_gpu_table()
    
    def _create_bitslice_map(self):
        """Create mapping from int values to bitslice info"""
        
        # Helper functions
        def int_to_bin(val, bits):
            min_val = -(1 << (bits - 1))
            max_val =  (1 << (bits - 1)) - 1
            if not (min_val <= val <= max_val):
                # Handle out of range values
                if val < min_val:
                    val = min_val
                elif val > max_val:
                    val = max_val
            if val < 0:
                val = (1 << bits) + val
            return format(val, f'0{bits}b')
        
        def bin_to_int(bin_str):
            if bin_str[0] == '1':  # negative
                return int(bin_str, 2) - (1 << len(bin_str))
            return int(bin_str, 2)
        
        def bin_to_uint(bin_str):
            return int(bin_str, 2)
        
        def uint_to_bin(val, bits):
            return format(val, f'0{bits}b')
        
        def get_bit_slices(val, bits, slice_width):
            bin_str = int_to_bin(val, bits)
            bin_str_rev = bin_str[::-1]  # reverse for LSB-first slicing
            slices = [bin_str_rev[i:i+slice_width][::-1] for i in range(0, bits, slice_width)]
            return slices
        
        def BADA_bit_slice(in_slc, slc_bw):
            slc_len = len(in_slc)
            cin = np.zeros(slc_len, dtype=int)
            cout = np.zeros(slc_len, dtype=int)
            out_slc = [""] * slc_len
            
            for i in range(slc_len-1):
                if i == 0:
                    cin[i] = 0
                else:
                    cin[i] = cout[i-1] or (bin_to_int(out_slc[i-1]) < 0)
                
                tmp_int = bin_to_uint(in_slc[i]) + cin[i]
                tmp_bin = uint_to_bin(tmp_int, slc_bw+1)
                out_slc[i] = tmp_bin[1:]
                cout[i] = int(tmp_bin[0])
            
            # Last slice
            i = slc_len - 1
            cin[i] = cout[i-1] or (bin_to_int(out_slc[i-1]) < 0)
            max_val = (1 << (slc_bw - 1)) - 1
            is_overflow = (bin_to_int(in_slc[i]) == max_val) and bool(cin[i])
            
            if is_overflow:
                out_slc[i] = in_slc[i]
                cout[i] = 0
            else:
                tmp_int = bin_to_uint(in_slc[i]) + cin[i]
                tmp_bin = uint_to_bin(tmp_int, slc_bw+1)
                out_slc[i] = tmp_bin[1:]
                cout[i] = int(tmp_bin[0])
            
            # Overflow slice
            overflow_slice = int_to_bin(is_overflow, slc_bw)
            out_slc.append(overflow_slice)
            
            return out_slc, is_overflow
        
        min_val = -(1 << (self.bits - 1))
        max_val =  (1 << (self.bits - 1)) - 1
        
        for val in range(min_val, max_val + 1):
            bin_slices = get_bit_slices(val, self.bits, self.slice_width)
            out_slices, has_overflow = BADA_bit_slice(bin_slices, self.slice_width)
            
            nonzero_count = 0
            reconstructed_val = 0
            
            for i, slice_bin in enumerate(out_slices[:-1]):
                slice_val = bin_to_int(slice_bin)
                if slice_val != 0:
                    nonzero_count += 1
                reconstructed_val += slice_val * (2 ** (i * self.slice_width))
            
            overflow_slice = out_slices[-1]
            overflow_val = bin_to_int(overflow_slice)
            
            if not self.discard_overflow and overflow_val != 0:
                nonzero_count += 1
                highest_power = ((len(out_slices) - 2) * self.slice_width)
                reconstructed_val += overflow_val * (2 ** highest_power)
            
            self.bitslice_map[val] = {
                'nonzero_slices': nonzero_count,
                'reconstructed_value': reconstructed_val,
                'has_overflow': has_overflow,
                # 'slices': out_slices,
                # 'original_value': clamped_val
            }
    
    def _create_gpu_table(self):
        """Create GPU lookup table"""
        min_val = -(1 << (self.bits - 1))
        max_val =  (1 << (self.bits - 1)) - 1
        
        # Create tensors for GPU lookup
        table_size = max_val - min_val + 1
        self.offset = -min_val  # Offset to map negative values to positive indices
        
        # Initialize with default values
        self.gpu_reconstructed = torch.zeros(table_size, dtype=torch.float32)
        self.gpu_nonzero_slices = torch.ones(table_size, dtype=torch.int32)
        self.gpu_has_overflow = torch.zeros(table_size, dtype=torch.bool)
        
        # Fill the tables
        for val, info in self.bitslice_map.items():
            idx = val + self.offset
            if 0 <= idx < table_size:
                self.gpu_reconstructed[idx] = info['reconstructed_value']
                self.gpu_nonzero_slices[idx] = info['nonzero_slices']
                self.gpu_has_overflow[idx] = info['has_overflow']
    
    def _move_to_device(self, device):
        """Move GPU tables to the specified device"""
        if self.device != device:
            self.gpu_reconstructed = self.gpu_reconstructed.to(device)
            self.gpu_nonzero_slices = self.gpu_nonzero_slices.to(device)
            self.gpu_has_overflow = self.gpu_has_overflow.to(device)
            self.device = device
    
    # def reconstruct(self, tensor: torch.Tensor) -> torch.Tensor:
    #     device = tensor.device
    #     values = tensor.detach().flatten().int()
        
    #     # GPU-based unique and inverse mapping
    #     unique_vals, inverse_indices = torch.unique(values, return_inverse=True)
        
    #     # Batch lookup on CPU (only for unique values)
    #     unique_vals_cpu = unique_vals.cpu().numpy()
    #     reconstructed_unique = torch.zeros_like(unique_vals, dtype=tensor.dtype)
        
    #     for i, val in enumerate(unique_vals_cpu):
    #         if int(val) in self.bitslice_map:
    #             reconstructed_unique[i] = self.bitslice_map[int(val)]['reconstructed_value']
    #         else:
    #             reconstructed_unique[i] = val
        
    #     # Map back to original tensor shape using inverse indices
    #     reconstructed_flat = reconstructed_unique[inverse_indices]
    #     return reconstructed_flat.reshape(tensor.shape).to(device)

    def reconstruct(self, tensor: torch.Tensor) -> torch.Tensor:
        # Ensure GPU tables are on the same device
        self._move_to_device(tensor.device)
        
        # Convert values to indices
        values = tensor.detach().int()
        indices = values + self.offset
        
        # Clamp indices to valid range
        indices = torch.clamp(indices, 0, len(self.gpu_reconstructed) - 1)
        
        # GPU lookup - no loops needed!
        reconstructed = self.gpu_reconstructed[indices]
        
        return reconstructed.reshape(tensor.shape)
    
    def get_bitslice_info(self, value: int) -> Dict:
        """Get bitslice info for analysis"""
        return self.bitslice_map.get(value, {
            'nonzero_slices': 1,
            'reconstructed_value': value,
            'has_overflow': False,
            # 'slices': ['000'],
            # 'original_value': value
        })


def create_bitslice_strategy(method: str, **kwargs) -> BitsliceStrategy:
    """Factory function to create bitslice strategies"""
    if method == "bada":
        return BADAStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown bitslice method: {method}")

if __name__ == "__main__":
    strategy1 = BADAStrategy(bits=9, slice_width=3, discard_overflow=False)
    strategy2 = BADAStrategy(bits=9, slice_width=3, discard_overflow=True)
    
    print("\n=== Testing value range with BADA strategy ===")
    
    test_values = range(-256, 256)
    for val in test_values:
        info1 = strategy1.get_bitslice_info(val)
        info2 = strategy2.get_bitslice_info(val)

        print(f"Value: {val:4d} | "
              f"Nonzero slices (keep): {info1['nonzero_slices']} | "
              f"Nonzero slices (discard): {info2['nonzero_slices']} | "
              f"Reconstructed (keep): {info1['reconstructed_value']:4d} | "
              f"Reconstructed (discard): {info2['reconstructed_value']:4d} |"
              f"Overflow: {info1['has_overflow']}")

    # strategy1 = BADAStrategy(bits=8, slice_width=4, discard_overflow=False)
    # strategy2 = BADAStrategy(bits=8, slice_width=4, discard_overflow=True)
    
    # print("\n=== Testing value range with BADA strategy ===")
    
    # test_values = range(-128, 128)
    # for val in test_values:
    #     info1 = strategy1.get_bitslice_info(val)
    #     info2 = strategy2.get_bitslice_info(val)

    #     print(f"Value: {val:4d} | "
    #           f"Nonzero slices (keep): {info1['nonzero_slices']} | "
    #           f"Nonzero slices (discard): {info2['nonzero_slices']} | "
    #           f"Reconstructed (keep): {info1['reconstructed_value']:4d} | "
    #           f"Reconstructed (discard): {info2['reconstructed_value']:4d} |"
    #           f"Overflow: {info1['has_overflow']}")
