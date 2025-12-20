#!/usr/bin/env python3
"""
YAMNet Audio Classification Inference Script

End-to-end audio inference pipeline for YAMNet on TI devices using TIDL.
This script processes an input audio file and runs inference using ONNX Runtime
with TIDLExecutionProvider for hardware acceleration.

Usage:
    python3 yamnet_infer_audio.py --audio-file samples/miaow_16k.wav
    python3 yamnet_infer_audio.py --audio-file samples/speech_whistling2.wav
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml

from yamnet_audio_processing import preprocess_audio_to_patches

def load_version_config():
    """Load and parse TIDL_VER from VERSION file."""
    version_file = Path(__file__).parent.parent.parent / 'VERSION'
    if not version_file.exists():
        raise FileNotFoundError(f"VERSION file not found at: {version_file}")

    config = dict(
        line.strip().split('=', 1) 
        for line in version_file.read_text().splitlines() 
        if line.strip() and not line.startswith('#')
    )

    if 'TIDL_VER' not in config:
        raise ValueError("TIDL_VER not found in VERSION file")

    return config['TIDL_VER']

TIDL_VER = load_version_config()

# Environment setup
if not (SOC := os.environ.get("SOC")):
    sys.exit("Error: SOC environment variable is not defined")

# Default model for inference
DEFAULT_MODEL = 'yamnet_combined'


def load_class_labels(yaml_path):
    """
    Load YAMNet class labels from YAML file.
        
    Returns:
        List of class names (521 AudioSet classes)
    """
    if not os.path.exists(yaml_path):
        print(f"Warning: Class labels file not found: {yaml_path}")
        return [f"Class_{i}" for i in range(521)]
    
    with open(yaml_path, 'r') as f:
        class_metadata = yaml.safe_load(f)
    
    class_names = [entry['name'] for entry in class_metadata]
    return class_names


def get_benchmark_output(benchmark):
    """Extract and compute benchmark metrics from TIDL benchmark data.
    
    Returns:
        tuple: (total_time_ms, proc_time_ms, copy_time_ms, ddr_read_mb, ddr_write_mb)
    """
    proc_time = copy_time = cp_in_time = cp_out_time = 0
    subgraph_ids = []
    
    # Extract subgraph IDs from benchmark keys
    for stat in benchmark.keys():
        if 'proc_start' in stat:
            subgraph_id = stat.split("ts:subgraph_")[1].split("_proc_start")[0]
            subgraph_ids.append(subgraph_id)
    
    # Accumulate timing metrics for each subgraph
    for subgraph_id in subgraph_ids:
        proc_time += (
            benchmark[f'ts:subgraph_{subgraph_id}_proc_end'] - 
            benchmark[f'ts:subgraph_{subgraph_id}_proc_start']
        )
        cp_in_time += (
            benchmark[f'ts:subgraph_{subgraph_id}_copy_in_end'] - 
            benchmark[f'ts:subgraph_{subgraph_id}_copy_in_start']
        )
        cp_out_time += (
            benchmark[f'ts:subgraph_{subgraph_id}_copy_out_end'] - 
            benchmark[f'ts:subgraph_{subgraph_id}_copy_out_start']
        )

    copy_time = cp_in_time + cp_out_time if len(subgraph_ids) == 1 else 0
    total_time = benchmark['ts:run_end'] - benchmark['ts:run_start']

    # Check if DDR stats are available (ARM platform)
    ddr_read = benchmark.get('ddr:read_end', 0) - benchmark.get('ddr:read_start', 0)
    ddr_write = benchmark.get('ddr:write_end', 0) - benchmark.get('ddr:write_start', 0)

    # Return in ms and MB
    return total_time / 1e6, proc_time / 1e6, copy_time / 1e6, ddr_read / 1e6, ddr_write / 1e6


def softmax(x):
    """Apply softmax function to input array."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()


def aggregate_predictions(predictions, class_names, top_k=10):
    """Aggregate predictions across all patches (time frames).
    
    Args:
        predictions: numpy array of shape [num_patches, num_classes]
        class_names: List of class names
        top_k: Number of top predictions to return
        
    Returns:
        dict: Dictionary with top predictions and statistics
    """
    # Average predictions across all patches
    mean_scores = np.mean(predictions, axis=0)
    
    # Apply softmax to get probabilities
    mean_scores_softmax = softmax(mean_scores)
    
    # Get top-k predictions
    top_indices = np.argsort(mean_scores_softmax)[::-1][:top_k]
    
    results = {
        'top_predictions': [],
        'num_patches': predictions.shape[0],
        'num_classes': predictions.shape[1]
    }
    
    for rank, idx in enumerate(top_indices, 1):
        results['top_predictions'].append({
            'rank': rank,
            'class_index': int(idx),
            'class_name': class_names[idx] if idx < len(class_names) else f"Class_{idx}",
            'score': float(mean_scores_softmax[idx])
        })
    
    return results


def generate_prediction_report(predictions, class_names):
    """Generate and display prediction report."""
    results = aggregate_predictions(predictions, class_names)
    
    print(f"\nAggregated predictions across {results['num_patches']} patches:")
    print(f"{'Rank':<6} {'Score':<10} {'Class'}")
    print("-" * 70)
    
    for pred in results['top_predictions']:
        print(f"{pred['rank']:<6} {pred['score']:<10.6f} {pred['class_name']}")
    
    return results


def generate_benchmark_report(sess, inference_time_ms, num_patches):
    """Generate and display benchmark report."""
    benchmark_dict = sess.get_TI_benchmark_data()
    total_time, subgraph_proc_time, copy_time, ddr_read, ddr_write = get_benchmark_output(benchmark_dict)
    
    # Calculate per-patch averages
    avg_inference_time = inference_time_ms / num_patches

    benchmark_results = {
        "total_time_ms": float(total_time),
        "processing_time_ms": float(subgraph_proc_time),
        "copy_time_ms": float(copy_time),
        "ddr_read_MBs": float(ddr_read),
        "ddr_write_MBs": float(ddr_write),
        "inference_time_ms": float(inference_time_ms),
        "avg_inference_time_ms": float(avg_inference_time)
    }

    # Performance metrics explanation:
    # - Inference time (measured): Python-level measurement including all overhead
    #   (data conversion, Python overhead, etc.) - typically larger than TIDL metrics
    # - TIDL total time: Internal TIDL measurement from execution start to completion
    # - TIDL processing time: Actual compute time on the hardware accelerator
    # - TIDL copy time: Data transfer time (copy in + copy out)
    print("\nTIDL Performance (per patch):")
    print(f"  Inference time (measured): {avg_inference_time:.2f} ms")
    print(f"  TIDL total time: {total_time:.2f} ms")
    print(f"  TIDL processing time: {subgraph_proc_time:.2f} ms")
    print(f"  TIDL copy time: {copy_time:.2f} ms")
    if ddr_read > 0 or ddr_write > 0:
        print(f"  DDR read: {ddr_read:.2f} MB/s")
        print(f"  DDR write: {ddr_write:.2f} MB/s")
    print(f"\nTotal inference time ({num_patches} patches): {inference_time_ms:.2f} ms")

    return benchmark_results


def run_inference(sess, patches, class_names):
    """Run YAMNet inference on preprocessed patches.
    
    Returns:
        tuple: (predictions, inference_time_ms) where predictions is numpy array 
               of shape [num_patches, 521] and inference_time_ms is total time in ms
    """
    input_details = sess.get_inputs()
    output_details = sess.get_outputs()
    
    input_name = input_details[0].name
    output_name = output_details[0].name
    
    print(f"\nModel input: {input_name}, shape: {input_details[0].shape}")
    print(f"Model output: {output_name}, shape: {output_details[0].shape}")
    print(f"Number of patches to process: {patches.shape[0]}")
    
    # Run inference on all patches
    all_predictions = []
    total_inference_time_ms = 0
    
    for i in range(patches.shape[0]):
        single_patch = patches[i:i+1]  # Keep batch dimension of 1
        
        # Run inference
        start_time = time.time()
        outputs = sess.run([output_name], {input_name: single_patch.astype(np.float32)})
        inference_time_ms = (time.time() - start_time) * 1000
        total_inference_time_ms += inference_time_ms
        
        all_predictions.append(outputs[0])
        
        print(f"  Processed patch {i+1}/{patches.shape[0]} ({inference_time_ms:.2f} ms)")
        
        # Print top 3 predictions for this patch
        patch_scores = outputs[0][0]  # Shape: (521,)
        top_indices = np.argsort(patch_scores)[::-1][:10]
        print(f"    Top predictions for patch {i+1}:")
        for rank, idx in enumerate(top_indices[:3], 1):
            class_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
            print(f"      {rank}. {class_name}: {patch_scores[idx]:.6f}")
    
    # Concatenate all predictions
    predictions = np.concatenate(all_predictions, axis=0)
    
    print(f"\nTotal inference time: {total_inference_time_ms:.2f} ms")
    print(f"Average time per patch: {total_inference_time_ms/patches.shape[0]:.2f} ms")
    print(f"Predictions shape: {predictions.shape}")
    
    return predictions, total_inference_time_ms


def generate_summary_report(prediction_results, benchmark_results, duration, 
                           preprocessing_time_ms, model_name=None, tensor_bits=None, 
                           audio_filename=None, top_k=10):
    """Generate standardized summary report for both CLI and notebook."""
    separator = "=" * 70
    print(separator)
    print("YAMNET INFERENCE SUMMARY")
    print(separator)
    
    if audio_filename:
        print(f"\nAudio File: {audio_filename}")
    if model_name and tensor_bits:
        print(f"Model: {model_name} (int{tensor_bits})")
    print(f"Audio Duration: {duration:.2f} seconds")
    print(f"Number of Patches: {prediction_results['num_patches']}")
    
    print(f"\n{separator}")
    print(f"TOP {top_k} PREDICTIONS (averaged across all patches):")
    print(f"{separator}")
    print(f"{'Rank':<6} {'Score':<10} {'Class'}")
    print("-" * 70)
    
    for pred in prediction_results['top_predictions'][:top_k]:
        print(f"{pred['rank']:<6} {pred['score']:<10.6f} {pred['class_name']}")
    
    # Use the larger value between inference_time_ms and processing_time_ms for RTF calculation
    process_time_ms = max(
        benchmark_results['inference_time_ms'], 
        benchmark_results['processing_time_ms']
    )
    
    print(f"\n{separator}")
    print("PERFORMANCE METRICS:")
    print(f"{separator}")
    print(f"Preprocessing: {preprocessing_time_ms:.2f} ms")
    print(f"Inference Time (all patches): {benchmark_results['inference_time_ms']:.2f} ms total | {prediction_results['num_patches']} patches")
    print(f"  - TIDL Processing Time: {benchmark_results['processing_time_ms']:.2f} ms")
    print(f"  - TIDL Copy Time: {benchmark_results['copy_time_ms']:.2f} ms")
    print(f"Inference Time (per patch): {benchmark_results['avg_inference_time_ms']:.2f} ms avg")
    print(f"  - TIDL Processing Time: {benchmark_results['processing_time_ms']/prediction_results['num_patches']:.2f} ms avg")
    print(f"  - TIDL Copy Time: {benchmark_results['copy_time_ms']/prediction_results['num_patches']:.2f} ms avg")
    print(f"RTF = {process_time_ms/1000/duration:.4f} ({process_time_ms/1000:.3f}s / {duration:.2f}s)")
    if benchmark_results['ddr_read_MBs'] > 0 or benchmark_results['ddr_write_MBs'] > 0:
        print(f"Memory Bandwidth: {benchmark_results['ddr_read_MBs']:.2f} MB/s read | {benchmark_results['ddr_write_MBs']:.2f} MB/s write")
    print(f"{separator}")


def generate_detailed_report(args, patches, predictions, prediction_results, 
                            benchmark_results, total_start_time, duration):
    """Generate detailed performance report."""
    total_pipeline_time = time.time() - total_start_time
    
    if total_pipeline_time > 0:
        inference_ratio = (benchmark_results["inference_time_ms"] / 1000) / total_pipeline_time * 100
    else:
        inference_ratio = 0
    
    preprocessing_ratio = 100 - inference_ratio
    
    # Use the larger value between inference_time_ms and processing_time_ms for RTF calculation
    process_time_ms = max(
        benchmark_results['inference_time_ms'], 
        benchmark_results['processing_time_ms']
    )
    rtf = process_time_ms / 1000 / duration

    separator = "=" * 70
    print(f"\n{separator}")
    print("DETAILED PERFORMANCE REPORT")
    print(separator)
    print(f"Model: {args.model} (int{args.tensor_bits})")
    print(f"Audio file: {os.path.basename(args.audio_file)}")
    print(f"Patches shape: {patches.shape}")
    print(f"Predictions shape: {predictions.shape}")

    total_pipeline_time_ms = total_pipeline_time * 1000
    inference_time_ms = benchmark_results['inference_time_ms']
    preprocessing_time_ms = total_pipeline_time_ms - inference_time_ms
    
    print("\nPipeline Time Breakdown:")
    print(f"  Audio duration: {duration:.2f} seconds")
    print(f"  Total pipeline time: {total_pipeline_time_ms:.2f} ms (100%)")
    print(f"  Preprocessing time: {preprocessing_time_ms:.2f} ms ({preprocessing_ratio:.1f}%)")
    print(f"  Inference time: {inference_time_ms:.2f} ms ({inference_ratio:.1f}%)")
    print(f"  Average inference per patch: {inference_time_ms/patches.shape[0]:.2f} ms")
    print(f"  Real-time factor (RTF): {rtf:.4f}")

    print("\nTop Recognized Classes:")
    for i, pred in enumerate(prediction_results['top_predictions'][:5]):
        print(f"  {i+1}. {pred['class_name']} (Class {pred['class_index']}): {pred['score']:.6f}")

    print(separator)


def main():
    """Main entry point for YAMNet audio inference pipeline."""
    parser = argparse.ArgumentParser(description='YAMNet Inference Pipeline')
    parser.add_argument('--audio-file', type=str, default='samples/miaow_16k.wav', help='Path to input WAV audio file (default: samples/miaow_16k.wav)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help=f'Model name (default: {DEFAULT_MODEL})')
    parser.add_argument('--tensor-bits', type=int, default=8, choices=[8, 16], help='Tensor bits for quantization (default: 8)')
    parser.add_argument('--debug-level', type=int, default=0, choices=range(4), help='Debug level (0-3) for TIDL execution (default: 0)')
    args = parser.parse_args()

    # Set up paths
    work_dir = Path(__file__).parent.parent.parent.resolve()
    base_artifacts_folder = work_dir / 'model_artifacts' / TIDL_VER / SOC
    models_base_path = work_dir / 'models' / 'onnx'
    class_labels_path = Path(__file__).parent / 'yamnet_class_map.yml'

    # Validate audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        sys.exit(f"Error: Audio file '{args.audio_file}' not found")

    # Set environment variables for performance stats
    os.environ["TIDL_RT_PERFSTATS"] = "1"
    if platform.machine() == 'aarch64':
        os.environ["TIDL_RT_DDR_STATS"] = "1"

    # Validate model exists
    model_path = models_base_path / f"{args.model}.onnx"
    if not model_path.exists():
        sys.exit(f"Error: Model file not found: {model_path}")

    # Load class labels
    print(f"Loading YAMNet class labels from: {class_labels_path}")
    class_names = load_class_labels(str(class_labels_path))
    print(f"Loaded {len(class_names)} class labels")

    # Process audio to extract patches
    separator = "=" * 70
    print(f"\n{separator}")
    print(f"Processing audio file: {args.audio_file}")
    print(separator)
    
    total_start_time = time.time()
    preprocess_start_time = time.time()
    patches, spectrogram, sr, duration = preprocess_audio_to_patches(args.audio_file)
    preprocessing_time_ms = (time.time() - preprocess_start_time) * 1000
    
    print("\nAudio preprocessing completed:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Patches shape: {patches.shape}")
    print(f"  Spectrogram shape: {spectrogram.shape}")
    print(f"  Preprocessing time: {preprocessing_time_ms:.2f} ms")

    # Set up inference session
    artifacts_folder = base_artifacts_folder / f"{args.model}_int{args.tensor_bits}"
    delegate_options = {
        "artifacts_folder": str(artifacts_folder),
        "debug_level": args.debug_level
    }
    
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    
    print(f"\n{separator}")
    print(f"Running inference on model: {args.model}")
    print(f"Artifacts folder: {artifacts_folder}")
    print(separator)

    # Create inference session and run
    execution_providers = ['TIDLExecutionProvider', 'CPUExecutionProvider']
    sess = None
    
    try:
        sess = ort.InferenceSession(
            str(model_path),
            providers=execution_providers,
            provider_options=[delegate_options, {}],
            sess_options=session_options
        )
        
        # Run inference
        predictions, inference_time_ms = run_inference(sess, patches, class_names)
        
        # Generate reports
        prediction_results = generate_prediction_report(predictions, class_names)
        benchmark_results = generate_benchmark_report(sess, inference_time_ms, patches.shape[0])

        # Generate standard summary report
        generate_summary_report(
            prediction_results, 
            benchmark_results, 
            duration, 
            preprocessing_time_ms,
            model_name=args.model,
            tensor_bits=args.tensor_bits,
            audio_filename=os.path.basename(args.audio_file)
        )
            
    except Exception as e:
        sys.exit(f"Error during inference: {e}")
    finally:
        if sess:
            del sess

    print("\nYAMNet audio inference completed successfully")


if __name__ == "__main__":
    main()
