import numpy as np
import math
import itertools
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import json
import pandas as pd
import re

@dataclass
class MMOperation:
    """Represents a single matrix multiplication operation with batch support"""
    name: str
    M: int
    K: int
    N: int
    B: int = 1  # Batch size for the operation workload
    
@dataclass
class Conv2DOperation:
    """Represents a 2D convolution operation with batch support"""
    name: str
    input_shape: Tuple[int, int, int, int]  # (batch, channels, height, width)
    kernel_size: Tuple[int, int]  # (kernel_height, kernel_width)
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int] = (0, 0)
    out_channels: int = 1
    
    @property
    def B(self) -> int:
        """Batch size for the operation workload"""
        return self.input_shape[0]
    
    def unfold_to_mm(self) -> MMOperation:
        """Unfolds the Conv2D operation into an equivalent matrix multiplication."""
        batch, in_channels, input_h, input_w = self.input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        
        # Calculate output dimensions
        output_h = (input_h + 2 * pad_h - kernel_h) // stride_h + 1
        output_w = (input_w + 2 * pad_w - kernel_w) // stride_w + 1
        
        # Matrix multiplication dimensions
        M = output_h * output_w  # Number of output positions per batch
        K = kernel_h * kernel_w * in_channels  # Size of each filter
        N = self.out_channels  # Number of output channels
        B = batch  # Batch size from input shape
        
        return MMOperation(
            name=f"{self.name}_unfolded",
            M=M,
            K=K,
            N=N,
            B=B
        )

Operation = Union[MMOperation, Conv2DOperation]
    
@dataclass
class HardwareConfig:
    """Hardware configuration for the GEMM engine"""
    TM: int
    TK: int
    TN: int
    batch_size: int = 1  # Hardware batch processing capability
    ibuf_size: Tuple[int, int] = (128, 256)
    wbuf_size: Tuple[int, int] = (256, 256)
    obuf_size: Tuple[int, int] = (128, 256)

@dataclass
class MemoryAccessStats:
    """Memory access statistics for a single MM operation"""
    operation_name: str
    dataflow_order: str
    operation_type: str = "MM"
    operation_batch_size: int = 1  # Operation workload batch size
    hardware_batch_size: int = 1   # Hardware batch processing capability
    X_DRAM_load: int = 0
    W_DRAM_load: int = 0
    Y_psum_DRAM_load: int = 0
    Y_psum_DRAM_store: int = 0
    Y_DRAM_store: int = 0
    
    @property
    def total_bytes(self) -> int:
        return (self.X_DRAM_load + self.W_DRAM_load + 
                self.Y_psum_DRAM_load + self.Y_psum_DRAM_store + 
                self.Y_DRAM_store)
    
    def to_dict(self) -> Dict:
        return {
            'operation_name': self.operation_name,
            'dataflow_order': self.dataflow_order,
            'operation_type': self.operation_type,
            'operation_batch_size': self.operation_batch_size,
            'hardware_batch_size': self.hardware_batch_size,
            'X_DRAM_load': self.X_DRAM_load,
            'W_DRAM_load': self.W_DRAM_load,
            'Y_psum_DRAM_load': self.Y_psum_DRAM_load,
            'Y_psum_DRAM_store': self.Y_psum_DRAM_store,
            'Y_DRAM_store': self.Y_DRAM_store,
            'total_bytes': self.total_bytes
        }

def quantize_matrix(matrix, scale, zero_point, bit_width=8):
    """Symmetrically quantizes a floating-point matrix to a specified integer bit-width."""
    min_int_val = -(2**(bit_width - 1))
    max_int_val = (2**(bit_width - 1)) - 1
    
    quantized_matrix = np.round(matrix / scale) + zero_point
    quantized_matrix = np.clip(quantized_matrix, min_int_val, max_int_val)
    
    return quantized_matrix.astype(np.int8)

def simulate_single_mm(mm_op: MMOperation, 
                      hw_config: HardwareConfig, 
                      dataflow_order: List[str],
                      operation_type: str = "MM",
                      skip_computation: bool = True,
                      verbose: bool = False) -> Tuple[MemoryAccessStats, bool]:
    """Simulates a single GEMM operation with specified dataflow order and batch support."""
    M, K, N, B = mm_op.M, mm_op.K, mm_op.N, mm_op.B
    TM, TK, TN = hw_config.TM, hw_config.TK, hw_config.TN
    hardware_batch_size = hw_config.batch_size
    
    if verbose:
        print(f"Simulating {operation_type} operation: {mm_op.name}")
        print(f"  Dimensions: A=({B},{M},{K}), B=({K},{N}), C=({B},{M},{N})")
        print(f"  Operation batch size: {B}")
        print(f"  Hardware batch size: {hardware_batch_size}")
        print(f"  Dataflow: {' -> '.join(dataflow_order)}")
    
    # Calculate tile counts
    nM = math.ceil(M / TM)
    nK = math.ceil(K / TK)
    nN = math.ceil(N / TN)
    nB = math.ceil(B / hardware_batch_size)  # Number of hardware batch iterations needed
    
    # Initialize memory access statistics
    stats = MemoryAccessStats(
        operation_name=mm_op.name,
        dataflow_order=' -> '.join(dataflow_order),
        operation_type=operation_type,
        operation_batch_size=B,
        hardware_batch_size=hardware_batch_size
    )
    
    # Element sizes in bytes
    bytes_per_elem = {'X': 1, 'W': 1, 'Y': 1, 'Y_psum': 3}
    
    # Create loop ranges
    loop_ranges = {'M': range(nM), 'K': range(nK), 'N': range(nN)}
    
    # Generate matrices for each batch only if computation is not skipped
    correctness_passed = True
    if not skip_computation:
        np.random.seed(42)
        # Create matrices for the operation batch size B
        X_batches = []
        Y_psum_batches = []
        Y_DRAM_batches = []
        Y_ref_batches = []
        Y_accum_tile_batches = []
        
        for b in range(B):
            X_batch = np.random.randint(-128, 128, size=(M, K), dtype=np.int8)
            X_batches.append(X_batch)
            Y_psum_batches.append(np.zeros((M, N), dtype=np.int64))
            Y_DRAM_batches.append(np.zeros((M, N), dtype=np.int8))
            Y_accum_tile_batches.append(np.zeros((M, N), dtype=np.int64))
        
        # Weight matrix is shared across batches
        W_DRAM = np.random.randint(-128, 128, size=(K, N), dtype=np.int8)
        
        # Reference computation for correctness check
        for b in range(B):
            Y_ref = np.matmul(X_batches[b].astype(np.int64), W_DRAM.astype(np.int64))
            Y_ref_batches.append(Y_ref)
        
        # Calculate quantization parameters from all batches
        all_refs = np.concatenate([y.flatten() for y in Y_ref_batches])
        scale = np.max(np.abs(all_refs)) / ((2**(8 - 1)) - 1)
        zero_point = 0
    
    # Simulate nested loops for the specified dataflow
    # We process the operation batch B in chunks of hardware_batch_size
    for batch_iter in range(nB):
        current_batch_start = batch_iter * hardware_batch_size
        current_batch_end = min((batch_iter + 1) * hardware_batch_size, B)
        current_hw_batch_size = current_batch_end - current_batch_start
        
        for i_outer in loop_ranges[dataflow_order[0]]:
            for i_middle in loop_ranges[dataflow_order[1]]:
                for i_inner in loop_ranges[dataflow_order[2]]:
                    
                    # Map loop indices back to m, k, n
                    current_indices = {
                        dataflow_order[0]: i_outer,
                        dataflow_order[1]: i_middle,
                        dataflow_order[2]: i_inner,
                    }
                    m = current_indices['M']
                    k = current_indices['K']
                    n = current_indices['N']
                    
                    # Define tile boundaries
                    start_m_tile, end_m_tile = m * TM, min((m + 1) * TM, M)
                    start_k_tile, end_k_tile = k * TK, min((k + 1) * TK, K)
                    start_n_tile, end_n_tile = n * TN, min((n + 1) * TN, N)
                    
                    # Determine if K is the innermost loop
                    is_k_innermost = (dataflow_order[2] == 'K')
                    
                    # Y Accumulation Logic - Load Y_psum if needed (multiplied by current_hw_batch_size)
                    if not is_k_innermost and k > 0:
                        stats.Y_psum_DRAM_load += ((end_m_tile - start_m_tile) * (end_n_tile - start_n_tile) * bytes_per_elem['Y_psum'] * current_hw_batch_size)
                    
                    # Load X tile from DRAM (reuse logic, multiplied by current_hw_batch_size)
                    if not (dataflow_order[2] == 'N' and n != 0):
                        stats.X_DRAM_load += ((end_m_tile - start_m_tile) * (end_k_tile - start_k_tile) * bytes_per_elem['X'] * current_hw_batch_size)
                    
                    # Load W tile from DRAM (shared across batches, no multiplication by batch size)
                    if not (dataflow_order[2] == 'M' and m != 0):
                        stats.W_DRAM_load += ((end_k_tile - start_k_tile) * (end_n_tile - start_n_tile) * bytes_per_elem['W'])
                    
                    # Perform actual computation if requested
                    if not skip_computation:
                        for b_offset in range(current_hw_batch_size):
                            b = current_batch_start + b_offset
                            if is_k_innermost:
                                if k == 0:
                                    Y_accum_tile_batches[b] = np.zeros((end_m_tile - start_m_tile, end_n_tile - start_n_tile), dtype=np.int64)
                            else:
                                if k == 0:
                                    Y_accum_tile_batches[b] = np.zeros((end_m_tile - start_m_tile, end_n_tile - start_n_tile), dtype=np.int64)
                                else:
                                    Y_accum_tile_batches[b] = Y_psum_batches[b][start_m_tile:end_m_tile, start_n_tile:end_n_tile].copy()
                            
                            X_GBUF = X_batches[b][start_m_tile:end_m_tile, start_k_tile:end_k_tile]
                            W_GBUF = W_DRAM[start_k_tile:end_k_tile, start_n_tile:end_n_tile]
                            Y_accum_tile_batches[b] += np.matmul(X_GBUF.astype(np.int64), W_GBUF.astype(np.int64))
                            
                            # Store intermediate results back to batch-specific arrays
                            if k == nK - 1:  # Last K iteration
                                Y_DRAM_batches[b][start_m_tile:end_m_tile, start_n_tile:end_n_tile] = quantize_matrix(Y_accum_tile_batches[b], scale, zero_point)
                            elif not is_k_innermost:
                                Y_psum_batches[b][start_m_tile:end_m_tile, start_n_tile:end_n_tile] = Y_accum_tile_batches[b]
                    
                    # Store results (multiplied by current_hw_batch_size for output)
                    if k == nK - 1:  # Last K iteration
                        stats.Y_DRAM_store      += ((end_m_tile - start_m_tile) * (end_n_tile - start_n_tile) * bytes_per_elem['Y'     ] * current_hw_batch_size)
                    elif not is_k_innermost:
                        stats.Y_psum_DRAM_store += ((end_m_tile - start_m_tile) * (end_n_tile - start_n_tile) * bytes_per_elem['Y_psum'] * current_hw_batch_size)
    
    # Check correctness if computation was performed
    if not skip_computation:
        try:
            for b in range(B):
                Y_ref_q = quantize_matrix(Y_ref_batches[b], scale, zero_point)
                diff = np.abs(Y_DRAM_batches[b] - Y_ref_q)
                max_diff = np.max(diff)
                if max_diff > 1:  # Allow for small quantization errors
                    correctness_passed = False
                    if verbose:
                        print(f"  ⚠️  Correctness check failed for batch {b}! Max difference: {max_diff}")
                    break
            if correctness_passed and verbose:
                print(f"  ✅ Correctness check passed for all {B} batches!")
        except Exception as e:
            correctness_passed = False
            if verbose:
                print(f"  ❌ Correctness check error: {e}")
    
    if verbose:
        print(f"  Total memory access: {stats.total_bytes:,} bytes")
    
    return stats, correctness_passed

def simulate_multi_mm_model(operations: List[Operation],
                           hw_config: HardwareConfig,
                           dataflow_order: List[str],
                           skip_computation: bool = True,
                           verbose: bool = True,
                           output_file: Optional[str] = None) -> List[MemoryAccessStats]:
    """Simulates memory access patterns for a model with multiple MM and Conv2D operations."""
    if verbose:
        print("=" * 80)
        print("MULTI-OPERATION GEMM SIMULATION")
        print("=" * 80)
        print(f"Hardware Config: TM={hw_config.TM}, TK={hw_config.TK}, TN={hw_config.TN}, HW_Batch={hw_config.batch_size}")
        print(f"Dataflow Order: {' -> '.join(dataflow_order)}")
        print(f"Total Operations: {len(operations)}")
        print(f"Skip Computation: {skip_computation}")
        print("=" * 80)
    
    all_stats = []
    
    for i, operation in enumerate(operations):
        if verbose:
            print(f"\n[{i+1}/{len(operations)}] Processing {operation.name}...")
        
        # Convert Conv2D to MM if needed
        if isinstance(operation, Conv2DOperation):
            mm_op = operation.unfold_to_mm()
            operation_type = "Conv2D"
            if verbose:
                print(f"  Conv2D unfolded to MM: B={mm_op.B}, M={mm_op.M}, K={mm_op.K}, N={mm_op.N}")
                print(f"  Original Conv2D: input={operation.input_shape}, "
                      f"kernel={operation.kernel_size}, stride={operation.stride}, "
                      f"padding={operation.padding}, out_channels={operation.out_channels}")
        else:
            mm_op = operation
            operation_type = "MM"
            if verbose:
                print(f"  MM dimensions: B={mm_op.B}, M={mm_op.M}, K={mm_op.K}, N={mm_op.N}")
        
        stats, _ = simulate_single_mm(
            mm_op=mm_op,
            hw_config=hw_config,
            dataflow_order=dataflow_order,
            operation_type=operation_type,
            skip_computation=skip_computation,
            verbose=verbose
        )
        
        all_stats.append(stats)
    
    # Print summary with Y_psum load and store included
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Operation':<55} {'Type':<8} {'Op_B':<5} {'HW_B':<5} {'Dimensions':<30} {'Total Bytes':<12} {'X Load':<10} {'W Load':<10} {'Y_psum Load':<12} {'Y_psum Store':<12} {'Y Store':<10}")
        print("-" * 185)
        
        total_bytes_all = 0
        for stats in all_stats:
            # Find original operation for dimensions
            orig_op = next(op for op in operations if op.name == stats.operation_name.replace('_unfolded', ''))
            if isinstance(orig_op, Conv2DOperation):
                mm_op = orig_op.unfold_to_mm()
            else:
                mm_op = orig_op
            
            dims_str = f"B={mm_op.B},M={mm_op.M},K={mm_op.K},N={mm_op.N}"
            print(f"{stats.operation_name:<55} {stats.operation_type:<8} {stats.operation_batch_size:<5} {stats.hardware_batch_size:<5} {dims_str:<30} {stats.total_bytes:<12,} "
                  f"{stats.X_DRAM_load:<10,} {stats.W_DRAM_load:<10,} {stats.Y_psum_DRAM_load:<12,} {stats.Y_psum_DRAM_store:<12,} {stats.Y_DRAM_store:<10,}")
            total_bytes_all += stats.total_bytes
        
        print("-" * 185)
        print(f"{'TOTAL':<55} {'':<8} {'':<5} {'':<5} {'':<30} {total_bytes_all:<12,}")
        print(f"\nTotal Memory Access: {total_bytes_all:,} bytes ({total_bytes_all / (1024**3):.2f} GB)")
    
    # Save results to file if requested
    if output_file:
        results = {
            'hardware_config': {
                'TM': hw_config.TM,
                'TK': hw_config.TK, 
                'TN': hw_config.TN,
                'batch_size': hw_config.batch_size
            },
            'dataflow_order': dataflow_order,
            'skip_computation': skip_computation,
            'operations': [stats.to_dict() for stats in all_stats],
            'total_memory_bytes': sum(stats.total_bytes for stats in all_stats)
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        if verbose:
            print(f"\nResults saved to: {output_file}")
    
    return all_stats

def compare_all_dataflows(operation: Operation, 
                         hw_config: HardwareConfig,
                         skip_computation: bool = True,
                         verbose: bool = True) -> Dict[str, MemoryAccessStats]:
    """Compares all 6 possible dataflow permutations for a single operation."""
    # Convert Conv2D to MM if needed
    if isinstance(operation, Conv2DOperation):
        mm_op = operation.unfold_to_mm()
        operation_type = "Conv2D"
        op_name = operation.name
        op_batch_size = operation.B
    else:
        mm_op = operation
        operation_type = "MM"
        op_name = operation.name
        op_batch_size = operation.B
    
    if verbose:
        print("=" * 80)
        print(f"DATAFLOW COMPARISON FOR {op_name.upper()} ({operation_type})")
        print("=" * 80)
        print(f"Dimensions: A=({mm_op.B},{mm_op.M},{mm_op.K}), B=({mm_op.K},{mm_op.N}), C=({mm_op.B},{mm_op.M},{mm_op.N})")
        if isinstance(operation, Conv2DOperation):
            print(f"Original Conv2D: input={operation.input_shape}, kernel={operation.kernel_size}")
        print(f"Operation batch size: {op_batch_size}")
        print(f"Hardware: TM={hw_config.TM}, TK={hw_config.TK}, TN={hw_config.TN}, HW_Batch={hw_config.batch_size}")
        print(f"Skip Computation: {skip_computation}")
        print("=" * 80)
    
    loop_dims = ['M', 'K', 'N']
    all_dataflows = list(itertools.permutations(loop_dims))
    results = {}
    
    pass_count = 0
    
    for i, dataflow_tuple in enumerate(all_dataflows):
        dataflow_order = list(dataflow_tuple)
        order_str = ' -> '.join(dataflow_order)
        
        if verbose:
            print(f"\n[{i+1}/6] Testing dataflow: {order_str}")
        
        stats, correctness_passed = simulate_single_mm(
            mm_op=mm_op,
            hw_config=hw_config,
            dataflow_order=dataflow_order,
            operation_type=operation_type,
            skip_computation=skip_computation,
            verbose=False
        )
        
        results[order_str] = stats
        
        if not skip_computation:
            if correctness_passed:
                pass_count += 1
                status = "✅ PASSED"
            else:
                status = "❌ FAILED"
        else:
            status = "⏭️  SKIPPED"
        
        if verbose:
            print(f"  Status: {status}")
            print(f"  Total Memory: {stats.total_bytes:,} bytes")
            print(f"  X Load: {stats.X_DRAM_load:,}, W Load: {stats.W_DRAM_load:,}, Y_psum Load: {stats.Y_psum_DRAM_load:,}, Y_psum Store: {stats.Y_psum_DRAM_store:,}, Y Store: {stats.Y_DRAM_store:,}")
    
    if verbose:
        print("\n" + "=" * 80)
        print("DATAFLOW COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'Dataflow':<15} {'Total Bytes':<12} {'X Load':<10} {'W Load':<10} {'Y_psum Load':<12} {'Y_psum Store':<12} {'Y Store':<10}")
        print("-" * 95)
        
        # Sort by total memory access
        sorted_results = sorted(results.items(), key=lambda x: x[1].total_bytes)
        
        for order_str, stats in sorted_results:
            print(f"{order_str:<15} {stats.total_bytes:<12,} {stats.X_DRAM_load:<10,} "
                  f"{stats.W_DRAM_load:<10,} {stats.Y_psum_DRAM_load:<12,} {stats.Y_psum_DRAM_store:<12,} {stats.Y_DRAM_store:<10,}")
        
        if not skip_computation:
            print(f"\nCorrectness: {pass_count}/6 dataflows passed")
            if pass_count == 6:
                print("🎉 All dataflows produced correct results!")
            else:
                print("⚠️  Some dataflows failed correctness check!")
        
        # Find best and worst dataflows
        best_dataflow = min(results.items(), key=lambda x: x[1].total_bytes)
        worst_dataflow = max(results.items(), key=lambda x: x[1].total_bytes)
        
        print(f"\n🏆 Best dataflow: {best_dataflow[0]} ({best_dataflow[1].total_bytes:,} bytes)")
        print(f"🔻 Worst dataflow: {worst_dataflow[0]} ({worst_dataflow[1].total_bytes:,} bytes)")
        
        memory_ratio = worst_dataflow[1].total_bytes / best_dataflow[1].total_bytes
        print(f"📊 Memory difference: {memory_ratio:.2f}x")
    
    return results


def create_sample_model(shape_dir=None) -> List[Operation]:
    """Creates a sample model with operations from CSV file or fallbacks."""
    
    if shape_dir:  # Fixed: should be if shape_dir exists
        try:
            operations = extract_operations_from_csv(shape_dir, max_operations=None)
            print(f"Loaded {len(operations)} operations from {shape_dir}")
            return operations
        except Exception as e:
            print(f"Error loading from {shape_dir}: {e}")
            print("Falling back to default operations")
    else:
        print("No CSV path provided, using fallback operations")
    
    # Fallback operations
    operations = [
        Conv2DOperation("fallback_conv", input_shape=(2, 3, 32, 32), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), out_channels=64),
        MMOperation("fallback_mm", M=256, K=512, N=256, B=2),
    ]
    
    return operations

def analyze_batch_efficiency(operation: Operation, hw_config: HardwareConfig, verbose: bool = True) -> Dict:
    """Analyzes batch processing efficiency for different hardware batch sizes."""
    if isinstance(operation, Conv2DOperation):
        mm_op = operation.unfold_to_mm()
        op_type = "Conv2D"
    else:
        mm_op = operation
        op_type = "MM"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"BATCH EFFICIENCY ANALYSIS: {operation.name} ({op_type})")
        print(f"{'='*60}")
        print(f"Operation batch size: {mm_op.B}")
        print(f"Operation dimensions: B={mm_op.B}, M={mm_op.M}, K={mm_op.K}, N={mm_op.N}")
    
    results = {}
    
    # Test different hardware batch sizes
    hw_batch_sizes = [1, 2, 4, 8, 16]
    dataflow_order = ['M', 'K', 'N']  # Use a standard dataflow
    
    if verbose:
        print(f"\n{'HW Batch':<10} {'Memory (MB)':<12} {'Efficiency':<12} {'Iterations':<12}")
        print("-" * 50)
    
    baseline_memory = None
    
    for hw_batch in hw_batch_sizes:
        if hw_batch > mm_op.B:
            continue  # Skip if hardware batch > operation batch
            
        test_hw_config = HardwareConfig(
            TM=hw_config.TM, 
            TK=hw_config.TK, 
            TN=hw_config.TN, 
            batch_size=hw_batch
        )
        
        stats, _ = simulate_single_mm(
            mm_op=mm_op,
            hw_config=test_hw_config,
            dataflow_order=dataflow_order,
            operation_type=op_type,
            skip_computation=True,
            verbose=False
        )
        
        memory_mb = stats.total_bytes / (1024 * 1024)
        iterations = math.ceil(mm_op.B / hw_batch)
        
        if baseline_memory is None:
            baseline_memory = memory_mb
            efficiency = 1.0
        else:
            efficiency = baseline_memory / memory_mb
        
        results[hw_batch] = {
            'memory_bytes': stats.total_bytes,
            'memory_mb': memory_mb,
            'efficiency': efficiency,
            'iterations': iterations,
            'stats': stats
        }
        
        if verbose:
            print(f"{hw_batch:<10} {memory_mb:<12.2f} {efficiency:<12.2f} {iterations:<12}")
    
    if verbose:
        # Find optimal hardware batch size
        optimal_hw_batch = max(results.keys(), key=lambda k: results[k]['efficiency'])
        print(f"\n🏆 Optimal hardware batch size: {optimal_hw_batch}")
        print(f"   Memory savings: {results[optimal_hw_batch]['efficiency']:.2f}x")
        print(f"   Total iterations: {results[optimal_hw_batch]['iterations']}")
    
    return results

def parse_shape_string(shape_str: str) -> List[int]:
    """Parse shape string like '[2, 4, 32, 32]' into list of integers."""
    clean_shape = shape_str.replace('[', '').replace(']', '').replace(' ', '')
    return [int(x) for x in clean_shape.split(',')]

def infer_conv2d_params(input_shape: List[int], weight_shape: List[int], output_shape: List[int]) -> Dict:
    """
    Infer stride and padding for Conv2D operations based on input/output shapes.
    
    Args:
        input_shape: [batch, in_channels, in_height, in_width]
        weight_shape: [out_channels, in_channels, kernel_height, kernel_width]
        output_shape: [batch, out_channels, out_height, out_width]
    
    Returns:
        Dict with stride and padding estimates
    """
    _, _, in_h, in_w = input_shape
    out_channels, _, k_h, k_w = weight_shape
    _, _, out_h, out_w = output_shape
    
    # Calculate stride based on input/output ratio
    stride_h = max(1, round(in_h / out_h)) if out_h > 0 else 1
    stride_w = max(1, round(in_w / out_w)) if out_w > 0 else 1
    
    # Common patterns for stride and padding
    if k_h == 3 and k_w == 3:
        if stride_h == 1 and stride_w == 1:
            stride = (1, 1)
            padding = (1, 1)
        elif stride_h == 2 and stride_w == 2:
            stride = (2, 2)
            padding = (1, 1)
        else:
            stride = (stride_h, stride_w)
            padding = (1, 1)
    elif k_h == 1 and k_w == 1:
        stride = (stride_h, stride_w)
        padding = (0, 0)
    elif k_h == 2 and k_w == 2:
        stride = (2, 2)
        padding = (0, 0)
    elif k_h == 7 and k_w == 7:
        stride = (stride_h, stride_w)
        padding = (3, 3)
    else:
        # Calculate padding needed
        pad_h = max(0, ((out_h - 1) * stride_h + k_h - in_h) // 2)
        pad_w = max(0, ((out_w - 1) * stride_w + k_w - in_w) // 2)
        stride = (stride_h, stride_w)
        padding = (pad_h, pad_w)
    
    return {'stride': stride, 'padding': padding}

def extract_operations_from_csv(csv_file: str, max_operations: Optional[int] = None) -> List:
    """
    Extract operations from CSV shape data.
    
    Args:
        csv_file: Path to CSV file
        max_operations: Maximum number of operations to extract
    
    Returns:
        List of Operation objects (MMOperation or Conv2DOperation)
    """
    df = pd.read_csv(csv_file)
    operations = []
    modules = df['module'].unique()
    
    if max_operations:
        modules = modules[:max_operations]
    
    for module in modules:
        module_data = df[df['module'] == module]
        
        # Find input, weight, and output tensors
        input_data  = module_data[module_data['tensor_type'] == 'input']
        weight_data = module_data[module_data['tensor_type'] == 'weight']
        output_data = module_data[module_data['tensor_type'] == 'output']
        
        if len(input_data) == 1 and len(weight_data) == 1 and len(output_data) == 1:
            input_shape  = parse_shape_string(input_data.iloc[0]['shape'])
            weight_shape = parse_shape_string(weight_data.iloc[0]['shape'])
            output_shape = parse_shape_string(output_data.iloc[0]['shape'])
            
            # Clean module name for operation name
            op_name = re.sub(r'[^\w]', '_', module)
            
            if len(input_shape) == 4 and len(weight_shape) == 4:
                # Conv2D operation
                conv_params = infer_conv2d_params(input_shape, weight_shape, output_shape)
                
                operations.append(Conv2DOperation(
                    name=op_name,
                    input_shape=tuple(input_shape),
                    kernel_size=(weight_shape[2], weight_shape[3]),
                    stride=conv_params['stride'],
                    padding=conv_params['padding'],
                    out_channels=weight_shape[0]
                ))
                
            elif len(input_shape) == 3 and len(weight_shape) == 2:
                # Linear operation (represented as MM)
                operations.append(MMOperation(
                    name=op_name,
                    M=input_shape[1],
                    K=input_shape[2],
                    N=weight_shape[0],
                    B=input_shape[0]
                ))
    
    return operations

if __name__ == "__main__":
    # Define hardware configuration with batch support
    hw_config = HardwareConfig(TM=128, TK=256, TN=256, batch_size=2)
    
    # Define dataflow order
    dataflow_order = ['M', 'N', 'K']
    
    # Create sample model operations
    operations = create_sample_model(shape_dir="output/shape/Final/class0/shape/shapes.csv")
    # operations = create_sample_model(shape_dir="output/shape/Final/round0/shape/shapes.csv")
    # operations = create_sample_model()    # use fallback
    
    # Run multi-operation simulation
    print("RUNNING MULTI-OPERATION MODEL SIMULATION")
    results = simulate_multi_mm_model(
        operations=operations,
        hw_config=hw_config, 
        dataflow_order=dataflow_order,
        skip_computation=True,
        verbose=True,
        output_file="simulation_results.json"
    )
    
    """
    # Example: Compare all 6 dataflow orders for a Conv2D operation with batch size 2
    print("\n" * 2)
    test_conv_op = Conv2DOperation("dataflow_test_conv", input_shape=(2, 64, 56, 56), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), out_channels=128)
    
    dataflow_comparison = compare_all_dataflows(
        operation=test_conv_op,
        hw_config=hw_config,
        skip_computation=False,
        verbose=True
    )
    
    # Example: Compare dataflows for a traditional MM operation with batch size 2
    print("\n" * 2)
    test_mm_op = MMOperation("dataflow_test_mm", M=256, K=4608, N=1152, B=2)
    
    mm_dataflow_comparison = compare_all_dataflows(
        operation=test_mm_op,
        hw_config=hw_config,
        skip_computation=True,
        verbose=True
    )
    
    # New: Batch efficiency analysis
    print("\n" * 2)
    print("RUNNING BATCH EFFICIENCY ANALYSIS")
    
    # Analyze different operations with various batch sizes
    test_operations = [
        MMOperation("small_batch_mm", M=256, K=512, N=256, B=2),
        MMOperation("large_batch_mm", M=128, K=256, N=128, B=8),
        Conv2DOperation("batch_conv", input_shape=(4, 32, 28, 28), kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), out_channels=64)
    ]
    
    for op in test_operations:
        analyze_batch_efficiency(op, hw_config, verbose=True)
    
    # Demonstration of mismatched batch sizes
    print("\n" + "="*80)
    print("MISMATCHED BATCH SIZE DEMONSTRATION")
    print("="*80)
    
    # Operation with batch=6, Hardware with batch=4
    large_batch_op = MMOperation("large_batch_test", M=128, K=256, N=128, B=6)
    hw_config_small = HardwareConfig(TM=64, TK=128, TN=128, batch_size=4)
    
    print(f"Operation batch size: {large_batch_op.B}")
    print(f"Hardware batch size: {hw_config_small.batch_size}")
    print(f"This will require {math.ceil(large_batch_op.B / hw_config_small.batch_size)} hardware batch iterations")
    
    stats, _ = simulate_single_mm(
        mm_op=large_batch_op,
        hw_config=hw_config_small,
        dataflow_order=['M', 'K', 'N'],
        operation_type="MM",
        skip_computation=True,
        verbose=True
    )
    """