#!/usr/bin/env python3
"""
YAMNet Audio Classification Inference Script

End-to-end audio inference pipeline for YAMNet on TI devices using TIDL.
This script processes an input audio file and runs inference using ONNX Runtime
with TIDLExecutionProvider for hardware acceleration.

Usage:
    python3 yamnet_infer_audio.py --audio-file samples/miaow_16k.wav
    python3 yamnet_infer_audio.py --audio-file samples/speech_whistling2.wav --detailed-report
"""

import os
import sys
import time
import numpy as np
import argparse
import yaml
import onnxruntime as ort
import platform
from pathlib import Path

# Import the audio processing functions
from yamnet_audio_processing import preprocess_audio_to_patches

# Import global configuration (TIDL version)
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import TIDL_VER

# Enable debugging only when needed
DEBUG = False

def debug_print(*args, **kwargs):
    """Print debug messages if DEBUG is enabled"""
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

# Environment setup
if not (SOC := os.environ.get("SOC")):
    sys.exit("Error: SOC environment variable is not defined")

# Default model for inference
DEFAULT_MODEL = 'yamnet_combined'


def load_class_labels(yaml_path):
    """
    Load YAMNet class labels from YAML file.
    
    Args:
        yaml_path: Path to yamnet_class_map.yml
        
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
    """Extract timing and bandwidth metrics from TIDL benchmark data"""
    proc_time = copy_time = cp_in_time = cp_out_time = 0
    subgraphIds = []
    
    for stat in benchmark.keys():
        if 'proc_start' in stat:
            value = stat.split("ts:subgraph_")
            value = value[1].split("_proc_start")
            subgraphIds.append(value[0])
    
    for subgraphId in subgraphIds:
        proc_time += (benchmark['ts:subgraph_'+subgraphId+'_proc_end'] - 
                     benchmark['ts:subgraph_'+subgraphId+'_proc_start'])
        cp_in_time += (benchmark['ts:subgraph_'+subgraphId+'_copy_in_end'] - 
                      benchmark['ts:subgraph_'+subgraphId+'_copy_in_start'])
        cp_out_time += (benchmark['ts:subgraph_'+subgraphId+'_copy_out_end'] - 
                       benchmark['ts:subgraph_'+subgraphId+'_copy_out_start'])

    copy_time = cp_in_time + cp_out_time if len(subgraphIds) == 1 else 0
    total_time = benchmark['ts:run_end'] - benchmark['ts:run_start']

    # Check if DDR stats are available (ARM platform)
    ddr_read = benchmark.get('ddr:read_end', 0) - benchmark.get('ddr:read_start', 0)
    ddr_write = benchmark.get('ddr:write_end', 0) - benchmark.get('ddr:write_start', 0)

    # return in ms and MB
    return total_time/1e6, proc_time/1e6, copy_time/1e6, ddr_read/1e6, ddr_write/1e6


def softmax(x, axis=-1):
    """Apply softmax function to input array"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def aggregate_predictions(predictions, class_names, top_k=10):
    """
    Aggregate predictions across all patches (time frames).
    
    Args:
        predictions: numpy array of shape [num_patches, num_classes]
        class_names: list of class names
        top_k: number of top predictions to return
        
    Returns:
        Dictionary with top predictions and statistics
    """
    # Average predictions across all patches
    mean_scores = np.mean(predictions, axis=0)
    
    # Get top-k predictions
    top_indices = np.argsort(mean_scores)[::-1][:top_k]
    
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
            'score': float(mean_scores[idx])
        })
    
    return results


def generate_prediction_report(predictions, class_names):
    """Generate and display prediction report"""
    results = aggregate_predictions(predictions, class_names)
    
    print(f"\nAggregated predictions across {results['num_patches']} patches:")
    print(f"{'Rank':<6} {'Score':<10} {'Class'}")
    print("-" * 70)
    
    for pred in results['top_predictions']:
        print(f"{pred['rank']:<6} {pred['score']:<10.6f} {pred['class_name']}")
    
    return results


def generate_benchmark_report(sess, inference_time_ms):
    """Generate and display benchmark report"""
    benchmark_dict = sess.get_TI_benchmark_data()
    total_time, sub_graphs_proc_time, copy_time, ddr_read, ddr_write = get_benchmark_output(benchmark_dict)

    benchmark_results = {
        "total_time_ms": float(total_time),
        "processing_time_ms": float(sub_graphs_proc_time),
        "copy_time_ms": float(copy_time),
        "ddr_read_MBs": float(ddr_read),
        "ddr_write_MBs": float(ddr_write),
        "inference_time_ms": float(inference_time_ms)
    }

    print(f"\nTIDL Performance Metrics:")
    print(f"  Total time: {total_time:.2f} ms")
    print(f"  Processing time: {sub_graphs_proc_time:.2f} ms")
    print(f"  Copy time: {copy_time:.2f} ms")
    if ddr_read > 0 or ddr_write > 0:
        print(f"  DDR read: {ddr_read:.2f} MB/s")
        print(f"  DDR write: {ddr_write:.2f} MB/s")

    return benchmark_results


def run_inference(sess, patches, class_names):
    """
    Run YAMNet inference on preprocessed patches.
    
    Args:
        sess: ONNX Runtime session
        patches: numpy array of shape [num_patches, 1, 96, 64]
        class_names: list of class names for displaying predictions
        
    Returns:
        predictions: numpy array of shape [num_patches, 521]
        inference_time_ms: total inference time in milliseconds
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
    total_inference_time = 0
    
    for i in range(patches.shape[0]):
        single_patch = patches[i:i+1]  # Keep batch dimension of 1
        
        # Run inference
        start_time = time.time()
        outputs = sess.run([output_name], {input_name: single_patch.astype(np.float32)})
        inference_time = (time.time() - start_time) * 1000  # ms
        total_inference_time += inference_time
        
        all_predictions.append(outputs[0])
        
        print(f"  Processed patch {i+1}/{patches.shape[0]} ({inference_time:.2f} ms)")
        
        # Print top 10 predictions for this patch
        patch_scores = outputs[0][0]  # Shape: (521,)
        top_indices = np.argsort(patch_scores)[::-1][:10]
        print(f"    Top predictions for patch {i+1}:")
        for rank, idx in enumerate(top_indices, 1):
            class_name = class_names[idx] if idx < len(class_names) else f"Class_{idx}"
            print(f"      {rank}. {class_name}: {patch_scores[idx]:.6f}")
    
    # Concatenate all predictions
    predictions = np.concatenate(all_predictions, axis=0)
    
    print(f"\nTotal inference time: {total_inference_time:.2f} ms")
    print(f"Average time per patch: {total_inference_time/patches.shape[0]:.2f} ms")
    print(f"Predictions shape: {predictions.shape}")
    
    return predictions, total_inference_time


def generate_detailed_report(args, patches, predictions, prediction_results, 
                            benchmark_results, total_start_time):
    """Generate detailed performance report"""
    total_pipeline_time = time.time() - total_start_time
    inference_ratio = (benchmark_results["inference_time_ms"] / 1000) / total_pipeline_time * 100 if total_pipeline_time > 0 else 0
    preprocessing_ratio = 100 - inference_ratio

    print("\n" + "="*70)
    print("DETAILED PERFORMANCE REPORT")
    print("="*70)
    print(f"Model: {args.model} (int{args.tensor_bits})")
    print(f"Audio file: {os.path.basename(args.audio_file)}")
    print(f"Patches shape: {patches.shape}")
    print(f"Predictions shape: {predictions.shape}")

    print("\nPipeline Time Breakdown:")
    print(f"  Total pipeline time: {total_pipeline_time:.2f} seconds (100%)")
    print(f"  Preprocessing time: {total_pipeline_time - (benchmark_results['inference_time_ms']/1000):.2f}s ({preprocessing_ratio:.1f}%)")
    print(f"  Inference time: {benchmark_results['inference_time_ms']/1000:.2f}s ({inference_ratio:.1f}%)")
    print(f"  Average inference per patch: {benchmark_results['inference_time_ms']/patches.shape[0]:.2f} ms")

    print("\nTop Recognized Classes:")
    for i, pred in enumerate(prediction_results['top_predictions'][:5]):
        print(f"  {i+1}. {pred['class_name']} (Class {pred['class_index']}): {pred['score']:.6f}")

    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='YAMNet End-to-End Audio Inference Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python3 yamnet_infer_audio.py --audio-file samples/miaow_16k.wav
  python3 yamnet_infer_audio.py --audio-file samples/speech_whistling2.wav --detailed-report
        '''
    )
    parser.add_argument('--audio-file', type=str, default='samples/miaow_16k.wav', help='Path to input WAV audio file (default: samples/miaow_16k.wav)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help=f'Model name (default: {DEFAULT_MODEL})')
    parser.add_argument('--tensor-bits', type=int, default=8, choices=[8, 16], help='Tensor bits for quantization (default: 8)')
    parser.add_argument('--debug-level', type=int, default=0, choices=range(4), help='Debug level (0-3) for TIDL execution (default: 0)')
    parser.add_argument('--detailed-report', action='store_true', help='Generate a detailed performance report')
    args = parser.parse_args()

    # Basic configuration
    WORK_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
    base_artifacts_folder = os.path.join(WORK_DIR, 'model_artifacts', TIDL_VER, SOC)
    models_base_path = os.path.join(WORK_DIR, 'models', 'onnx')
    
    # Path to class labels
    script_dir = Path(__file__).parent
    class_labels_path = script_dir / 'yamnet_class_map.yml'

    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found")
        sys.exit(1)

    # Set environment variables for performance stats
    os.environ["TIDL_RT_PERFSTATS"] = "1"
    if platform.machine() == 'aarch64':
        os.environ["TIDL_RT_DDR_STATS"] = "1"

    # Setup model path
    model_path = os.path.join(models_base_path, f"{args.model}.onnx")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    # Load class labels
    print(f"Loading YAMNet class labels from: {class_labels_path}")
    class_names = load_class_labels(str(class_labels_path))
    print(f"Loaded {len(class_names)} class labels")

    # Process audio to extract patches
    print(f"\n{'='*70}")
    print(f"Processing audio file: {args.audio_file}")
    print(f"{'='*70}")
    
    total_start_time = time.time()
    start_time = time.time()
    patches, spectrogram, sr, duration = preprocess_audio_to_patches(args.audio_file)
    preprocess_time = time.time() - start_time
    
    print(f"\nAudio preprocessing completed:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Patches shape: {patches.shape}")
    print(f"  Spectrogram shape: {spectrogram.shape}")
    print(f"  Preprocessing time: {preprocess_time*1000:.2f} ms")

    # Set up inference session
    artifacts_folder = os.path.join(base_artifacts_folder, f"{args.model}_int{args.tensor_bits}")
    delegate_options = {
        "artifacts_folder": artifacts_folder,
        "debug_level": args.debug_level
    }
    
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    
    print(f"\n{'='*70}")
    print(f"Running inference on model: {args.model}")
    print(f"Artifacts folder: {artifacts_folder}")
    print(f"{'='*70}")

    # Create inference session and run
    EP_list = ['TIDLExecutionProvider', 'CPUExecutionProvider']
    sess = None
    
    try:
        sess = ort.InferenceSession(
            model_path,
            providers=EP_list,
            provider_options=[delegate_options, {}],
            sess_options=session_options
        )
        
        # Run inference
        predictions, inference_time_ms = run_inference(sess, patches, class_names)
        
        # Generate reports
        prediction_results = generate_prediction_report(predictions, class_names)
        benchmark_results = generate_benchmark_report(sess, inference_time_ms)

        if args.detailed_report:
            generate_detailed_report(
                args, patches, predictions, prediction_results, 
                benchmark_results, total_start_time
            )
            
    except Exception as e:
        print(f"\nError during inference: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        if sess:
            del sess

    total_time = time.time() - total_start_time
    print(f"\n{'='*70}")
    print(f"Performance Summary:")
    print(f"  Preprocessing time: {preprocess_time*1000:.2f} ms")
    print(f"  Inference time: {benchmark_results['inference_time_ms']:.2f} ms")
    print(f"  Total pipeline time: {total_time:.2f} seconds")
    print("YAMNet audio inference completed successfully")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
