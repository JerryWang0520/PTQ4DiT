import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any


class BitsliceStrategy(ABC):
    def __init__(self, bits=9, slice_width=3):
        self.bits = bits
        self.slice_width = slice_width
        self.device = None
        self.bitslice_map = {}
        self.gpu_reconstructed = None
        self.gpu_nonzero_slices = None
        self.gpu_has_overflow = None
        self.offset = None
        self.discard_overflow = False
        self.bitslice_clamp = False
    
    @abstractmethod
    def reconstruct(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reconstruct tensor using bitslice method"""
        pass
    
    def _create_gpu_table(self):
        """Create GPU lookup tables from bitslice_map"""
        min_val = -(1 << (self.bits - 1))
        max_val =  (1 << (self.bits - 1)) - 1
        
        table_size = max_val - min_val + 1
        self.offset = -min_val
        
        self.gpu_reconstructed  = torch.zeros(table_size, dtype=torch.float32)
        self.gpu_nonzero_slices = torch.ones (table_size, dtype=torch.int32)
        self.gpu_has_overflow   = torch.zeros(table_size, dtype=torch.bool)
        
        for val, info in self.bitslice_map.items():
            idx = val + self.offset
            if 0 <= idx < table_size:
                self.gpu_reconstructed [idx] = info['reconstructed_value']
                self.gpu_nonzero_slices[idx] = info['nonzero_slices']
                self.gpu_has_overflow  [idx] = info['has_overflow']
    
    def _move_to_device(self, device):
        """Move GPU tables to the specified device"""
        if self.device != device:
            self.gpu_reconstructed  = self.gpu_reconstructed.to(device)
            self.gpu_nonzero_slices = self.gpu_nonzero_slices.to(device)
            self.gpu_has_overflow   = self.gpu_has_overflow.to(device)
            self.device = device
    
    @staticmethod
    def _int_to_bin(val, bits):
        min_val = -(1 << (bits - 1))
        max_val =  (1 << (bits - 1)) - 1
        if val < min_val:
            val = min_val
        elif val > max_val:
            val = max_val
        if val < 0:
            val = (1 << bits) + val
        return format(val, f'0{bits}b')
    
    @staticmethod
    def _bin_to_int(bin_str):
        if bin_str[0] == '1':
            return int(bin_str, 2) - (1 << len(bin_str))
        return int(bin_str, 2)
    
    @staticmethod
    def _bin_to_uint(bin_str):
        return int(bin_str, 2)
    
    @staticmethod
    def _uint_to_bin(val, bits):
        return format(val, f'0{bits}b')
    
    def get_bit_slices(self, val, bits, slice_width):
        bin_str = self._int_to_bin(val, bits)
        bin_str_rev = bin_str[::-1]  # reverse for LSB-first slicing
        slices = [bin_str_rev[i:i+slice_width][::-1] for i in range(0, bits, slice_width)]
        return slices
    
    def reconstruct(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reconstruct tensor using bitslice method"""
        self._move_to_device(tensor.device)
        values = tensor.detach().int()
        indices = values + self.offset
        indices = torch.clamp(indices, 0, len(self.gpu_reconstructed) - 1)
        reconstructed = self.gpu_reconstructed[indices]
        return reconstructed.reshape(tensor.shape)
    
    def get_bitslice_info(self, value: int) -> Dict:
        """Get bitslice info for analysis"""
        return self.bitslice_map.get(value, {
            'nonzero_slices': 1,
            'reconstructed_value': value,
            'has_overflow': False,
            'slices': None,
        })


class BADAStrategy(BitsliceStrategy):
    def __init__(self, bits=9, slice_width=3, discard_overflow=False, bitslice_clamp=False):
        super().__init__(bits, slice_width)
        self.discard_overflow = discard_overflow
        self.bitslice_clamp = bitslice_clamp
        self._create_bitslice_map()
        self._create_gpu_table()
    
    def _create_bitslice_map(self):
        def BADA_bit_slice(in_slc, slc_bw):
            slc_len = len(in_slc)
            cin = np.zeros(slc_len, dtype=int)
            cout = np.zeros(slc_len, dtype=int)
            out_slc = [""] * slc_len
            
            for i in range(slc_len-1):
                if i == 0:
                    cin[i] = 0
                else:
                    cin[i] = cout[i-1] or (self._bin_to_int(out_slc[i-1]) < 0)
                
                tmp_int = self._bin_to_uint(in_slc[i]) + cin[i]
                tmp_bin = self._uint_to_bin(tmp_int, slc_bw+1)
                out_slc[i] = tmp_bin[1:]
                cout[i] = int(tmp_bin[0])
            
            # Last slice
            i = slc_len - 1
            cin[i] = cout[i-1] or (self._bin_to_int(out_slc[i-1]) < 0)
            max_val = (1 << (slc_bw - 1)) - 1
            is_overflow = (self._bin_to_int(in_slc[i]) == max_val) and bool(cin[i])
            
            if is_overflow:
                out_slc[i] = in_slc[i]
                cout[i] = 0
            else:
                tmp_int = self._bin_to_uint(in_slc[i]) + cin[i]
                tmp_bin = self._uint_to_bin(tmp_int, slc_bw+1)
                out_slc[i] = tmp_bin[1:]
                cout[i] = int(tmp_bin[0])
            
            # Overflow slice
            overflow_slice = self._int_to_bin(is_overflow, slc_bw)
            out_slc.append(overflow_slice)
            
            return out_slc, is_overflow
        
        min_val = -(1 << (self.bits - 1))
        max_val =  (1 << (self.bits - 1)) - 1
        
        for val in range(min_val, max_val + 1):
            slices = self.get_bit_slices(val, self.bits, self.slice_width)
            out_slices, has_overflow = BADA_bit_slice(slices, self.slice_width)
            
            nonzero_count = 0
            reconstructed_val = 0
            
            for i, slice_bin in enumerate(out_slices[:-1]):
                slice_val = self._bin_to_int(slice_bin)
                if slice_val != 0:
                    nonzero_count += 1
                reconstructed_val += slice_val * (2 ** (i * self.slice_width))
            
            overflow_slice = out_slices[-1]
            overflow_val = self._bin_to_int(overflow_slice)
            
            if not self.discard_overflow and overflow_val != 0:
                nonzero_count += 1
                highest_power = ((len(out_slices) - 2) * self.slice_width)
                reconstructed_val += overflow_val * (2 ** highest_power)
            
            self.bitslice_map[val] = {
                'nonzero_slices': nonzero_count,
                'reconstructed_value': reconstructed_val,
                'has_overflow': has_overflow,
                'slices': out_slices,
            }


class NaiveStrategy(BitsliceStrategy):
    def __init__(self, bits=9, slice_width=3):
        super().__init__(bits, slice_width)
        
        # Calculate extended bits for uniform slicing
        self.num_slices = (bits + slice_width - 1) // slice_width
        self.extended_bits = self.num_slices * slice_width
        
        self._create_bitslice_map()
        self._create_gpu_table()
    
    def _create_bitslice_map(self):
        min_val = -(1 << (self.bits - 1))
        max_val =  (1 << (self.bits - 1)) - 1
        
        for val in range(min_val, max_val + 1):
            slices = self.get_bit_slices(val, self.extended_bits, self.slice_width)

            nonzero_count = 0
            reconstructed_val = 0
            
            for i, slice_bin in enumerate(slices):
                if i == len(slices) - 1:
                    slice_val = self._bin_to_int(slice_bin)   # Signed for HOS
                else:
                    slice_val = self._bin_to_uint(slice_bin)  # Unsigned for all other slices
                    
                if slice_val != 0:
                    nonzero_count += 1
                reconstructed_val += slice_val * (2 ** (i * self.slice_width))
            
            self.bitslice_map[val] = {
                'nonzero_slices': nonzero_count,
                'reconstructed_value': reconstructed_val,
                'has_overflow': False,
                'slices': slices,
            }


class SibiaStrategy(BitsliceStrategy):
    def __init__(self, bits=9, slice_width=5):
        super().__init__(bits, slice_width)
        
        # Calculate extended bits for uniform slicing
        self.num_slices = ((bits - 1) // (slice_width - 1))
        self.extended_bits = slice_width + (self.num_slices - 1) * (slice_width - 1)

        self._create_bitslice_map()
        self._create_gpu_table()
    
    def _create_bitslice_map(self):
        min_val = -(1 << (self.bits - 1))
        max_val =  (1 << (self.bits - 1)) - 1
        
        for val in range(min_val, max_val + 1):
            sign_bit = "1" if val < 0 else "0"
            slices = self.get_bit_slices(val, self.extended_bits, self.slice_width-1)
            slices.pop()

            modified_slices = []
            for i, slice_bin in enumerate(slices):
                if i == len(slices) - 1:
                    modified_slice = sign_bit + slice_bin
                else:
                    modified_slice = "0" + slice_bin
                modified_slices.append(modified_slice)
            slices = modified_slices

            modified_slices = []
            for i, slice_bin in enumerate(slices):
                if i == 0:
                    slice_int = self._bin_to_int(slice_bin)
                    if sign_bit == "1":
                        slice_int = slice_int - (1 << (self.slice_width - 1))
                    modified_slice = self._int_to_bin(slice_int, self.slice_width)
                elif i == len(slices) - 1:
                    slice_int = self._bin_to_int(slice_bin)
                    if sign_bit == "1":
                        slice_int = slice_int + 1
                    modified_slice = self._int_to_bin(slice_int, self.slice_width)
                else:
                    slice_int = self._bin_to_int(slice_bin)
                    if sign_bit == "1":
                        slice_int = slice_int - (1 << (self.slice_width - 1)) + 1
                    modified_slice = self._int_to_bin(slice_int, self.slice_width)
                modified_slices.append(modified_slice)
            slices = modified_slices

            nonzero_count = 0
            reconstructed_val = 0

            for i, slice_bin in enumerate(slices):
                slice_val = self._bin_to_int(slice_bin)                    
                if slice_val != 0:
                    nonzero_count += 1
                reconstructed_val += slice_val * (2 ** (i * (self.slice_width-1)))
            
            self.bitslice_map[val] = {
                'nonzero_slices': nonzero_count,
                'reconstructed_value': reconstructed_val,
                'has_overflow': False,
                'slices': slices,
            }


def create_bitslice_strategy(method: str, **kwargs) -> BitsliceStrategy:
    common_params = {'bits', 'slice_width'}

    if method == "bada":
        return BADAStrategy(**kwargs)
    elif method == "naive":
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in common_params}
        return NaiveStrategy(**filtered_kwargs)
    elif method == "sibia":
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in common_params}
        return SibiaStrategy(**filtered_kwargs)
    else:
        raise ValueError(f"Unknown bitslice method: {method}")

if __name__ == "__main__":
    # BADA sint9 -> 3*sint3 + overflow_slice
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
        assert val == info1['reconstructed_value'], f"reconstruction value mismatch"

    # BADA sint8 -> 2*sint4 + overflow_slice
    strategy1 = BADAStrategy(bits=8, slice_width=4, discard_overflow=False)
    strategy2 = BADAStrategy(bits=8, slice_width=4, discard_overflow=True)
    
    print("\n=== Testing value range with BADA strategy ===")
    
    test_values = range(-128, 128)
    for val in test_values:
        info1 = strategy1.get_bitslice_info(val)
        info2 = strategy2.get_bitslice_info(val)

        print(f"Value: {val:4d} | "
              f"Nonzero slices (keep): {info1['nonzero_slices']} | "
              f"Nonzero slices (discard): {info2['nonzero_slices']} | "
              f"Reconstructed (keep): {info1['reconstructed_value']:4d} | "
              f"Reconstructed (discard): {info2['reconstructed_value']:4d} |"
              f"Overflow: {info1['has_overflow']}")
        assert val == info1['reconstructed_value'], f"reconstruction value mismatch"

    # Naive sint9 -> sint3 + uint3 + uint3 -> sint4 MAC
    strategy1 = NaiveStrategy(bits=9, slice_width=3)
    
    print("\n=== Testing value range with Naive strategy ===")
    
    test_values = range(-256, 256)
    for val in test_values:
        info1 = strategy1.get_bitslice_info(val)

        print(f"Value: {val:4d} | "
              f"Nonzero slices (keep): {info1['nonzero_slices']} | "
              f"Reconstructed (keep): {info1['reconstructed_value']:4d} | "
              f"Overflow: {info1['has_overflow']} | "
              f"Slices (HOS <-> LOS): {info1['slices'][::-1]}")
        assert val == info1['reconstructed_value'], f"reconstruction value mismatch"

    # Sibia sint9 -> 2*sint5
    strategy1 = SibiaStrategy(bits=9, slice_width=5)
    
    print("\n=== Testing value range with Sibia strategy ===")
    
    test_values = range(-256, 256)
    # test_values = range(-512, 512)
    for val in test_values:
        info1 = strategy1.get_bitslice_info(val)

        print(f"Value: {val:4d} | "
              f"Nonzero slices (keep): {info1['nonzero_slices']} | "
              f"Reconstructed (keep): {info1['reconstructed_value']:4d} | "
              f"Overflow): {info1['has_overflow']} | "
              f"Slices (HOS <-> LOS): {info1['slices'][::-1]}")
        assert val == info1['reconstructed_value'], f"reconstruction value mismatch"
