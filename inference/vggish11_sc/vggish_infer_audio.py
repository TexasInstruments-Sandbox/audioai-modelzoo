#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import argparse
import torch
import torchaudio
import onnxruntime as ort
import platform
from datetime import datetime

# Import the audio processing functions
try:
    from audio_processing import preprocess_audio_waveform, log_mel_spectrogram
except ImportError:
    print("Warning: Could not import audio_processing. Ensure it's in the Python path.")
    sys.exit(1)

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
DEFAULT_MODEL = 'vggish11_20250324-1807_ptq'

# UrbanSound8K class label mapping built-in
CLASS_LABELS = {
    "0": "Air conditioner",
    "1": "Car horn",
    "2": "Children playing",
    "3": "Dog bark",
    "4": "Drilling",
    "5": "Engine idling",
    "6": "Gun shot",
    "7": "Jackhammer",
    "8": "Siren",
    "9": "Street music"
}

# Audio processing configuration
AUDIO_CONFIG_DEFAULT = {
    'dataset': {
        'sample_rate': 16000,
        'duration': 4.0,
        'n_fft': 1024,
        'hop_length': 512,
        'n_mels': 64
    }
}

def get_benchmark_output(benchmark):
    proc_time = copy_time = cp_in_time = cp_out_time = 0
    subgraphIds = []
    for stat in benchmark.keys():
        if 'proc_start' in stat:
            value = stat.split("ts:subgraph_")
            value = value[1].split("_proc_start")
            subgraphIds.append(value[0])
    for subgraphId in subgraphIds:
        proc_time += (benchmark['ts:subgraph_'+subgraphId+'_proc_end'] - benchmark['ts:subgraph_'+subgraphId+'_proc_start'])
        cp_in_time += (benchmark['ts:subgraph_'+subgraphId+'_copy_in_end'] - benchmark['ts:subgraph_'+subgraphId+'_copy_in_start'])
        cp_out_time += (benchmark['ts:subgraph_'+subgraphId+'_copy_out_end'] - benchmark['ts:subgraph_'+subgraphId+'_copy_out_start'])

    copy_time = cp_in_time + cp_out_time if len(subgraphIds) == 1 else 0
    total_time = benchmark['ts:run_end'] - benchmark['ts:run_start']

    # Check if DDR stats are available (ARM platform)
    ddr_read = benchmark.get('ddr:read_end', 0) - benchmark.get('ddr:read_start', 0)
    ddr_write = benchmark.get('ddr:write_end', 0) - benchmark.get('ddr:write_start', 0)

    # return in ms and MB
    return total_time/1e6, proc_time/1e6, copy_time/1e6, ddr_read/1e6, ddr_write/1e6

def preprocess_audio_to_features(audio_path, config):
    """Process audio file to extract features suitable for model input"""
    print(f"\nProcessing audio file: {audio_path}")

    # Step 1: Preprocess audio waveform
    start_time = time.time()
    waveform = preprocess_audio_waveform(
        audio_path,
        sample_rate=config['dataset']['sample_rate'],
        duration=config['dataset']['duration']
    )
    preprocess_time = time.time() - start_time
    print(f"  Waveform shape: {waveform.shape}, Preprocessing time: {preprocess_time:.4f}s")

    # Step 2: Extract log mel spectrogram features
    start_time = time.time()
    mel_spec = log_mel_spectrogram(waveform, config)
    feature_time = time.time() - start_time
    print(f"  Log mel spectrogram shape: {mel_spec.shape}, Extraction time: {feature_time:.4f}s")

    return mel_spec.cpu().numpy()

def format_class_name(idx):
    """Format class index with name from built-in class labels"""
    str_idx = str(idx)
    return f"Class {idx} ({CLASS_LABELS[str_idx]})" if str_idx in CLASS_LABELS else f"Class {idx}"

def softmax(x):
    """Apply softmax function to input array"""
    exp_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return exp_x / exp_x.sum()

def generate_prediction_report(result_softmax, sorted_indices):
    """Generate and display prediction report with visualizations"""
    top_indices = sorted_indices[:5]  # get indices of top 5 predictions
    prediction_results = {"top_predictions": [], "all_predictions": []}

    # Process top predictions with visualization
    bar_length = 40
    max_confidence = result_softmax[top_indices[0]] if top_indices.size > 0 else 1.0 # Avoid division by zero if no predictions

    print("\nTop predictions (softmax probabilities):")
    for i, idx in enumerate(top_indices):
        confidence = result_softmax[idx]
        class_info = format_class_name(idx)

        print(f"  {i+1}. {class_info}: {confidence:.6f}")
        prediction_results["top_predictions"].append({
            "rank": i+1, "class_index": int(idx),
            "class_name": CLASS_LABELS.get(str(idx), "Unknown"),
            "confidence": float(confidence)
        })

    # Store all predictions in the dictionary
    for i, idx in enumerate(sorted_indices):
        prediction_results["all_predictions"].append({
            "rank": i+1, "class_index": int(idx),
            "class_name": CLASS_LABELS.get(str(idx), "Unknown"),
            "confidence": float(result_softmax[idx])
        })

    return prediction_results

def generate_benchmark_report(sess, inference_time):
    """Generate and display benchmark report"""
    benchmark_dict = sess.get_TI_benchmark_data()
    total_time, sub_graphs_proc_time, copy_time, ddr_read, ddr_write = get_benchmark_output(benchmark_dict)

    # Store and display benchmark results
    benchmark_results = {
        "total_time_ms": float(total_time),
        "processing_time_ms": float(sub_graphs_proc_time),
        "copy_time_ms": float(copy_time),
        "ddr_read_MBs": float(ddr_read),
        "ddr_write_MBs": float(ddr_write),
        "inference_time_ms": float(inference_time)
    }

    print(f"\nTIDL Performance:")
    print(f"  Total time: {total_time:.2f} ms")
    print(f"  Processing time: {sub_graphs_proc_time:.2f} ms")
    print(f"  Copy time: {copy_time:.2f} ms")
    if ddr_read > 0 or ddr_write > 0:
        print(f"  DDR read: {ddr_read:.2f} MB/s, DDR write: {ddr_write:.2f} MB/s")

    return benchmark_results

def generate_detailed_report(args, features, outputs, prediction_results, benchmark_results, total_start_time):
    """Generate detailed performance report"""
    total_pipeline_time = time.time() - total_start_time
    inference_ratio = (benchmark_results["inference_time_ms"] / 1000) / total_pipeline_time * 100 if total_pipeline_time > 0 else 0
    preprocessing_ratio = 100 - inference_ratio

    print("\n" + "="*50)
    print("DETAILED PERFORMANCE REPORT")
    print("="*50)
    print(f"Model: {args.model} (int{args.tensor_bits})")
    print(f"Audio file: {os.path.basename(args.audio_file)}")
    print(f"Feature shape: {features.shape}, Output shape: {outputs[0].shape}")

    print("\nPipeline Time Breakdown:")
    print(f"  Total pipeline time: {total_pipeline_time:.2f} seconds (100%)")
    print(f"  Preprocessing time: {total_pipeline_time - (benchmark_results['inference_time_ms']/1000):.2f}s ({preprocessing_ratio:.1f}%)")
    print(f"  Inference time: {benchmark_results['inference_time_ms']/1000:.2f}s ({inference_ratio:.1f}%)")

    print("\nClass Distribution Analysis:")
    print("  Top 3 recognized classes:")
    for i in range(min(3, len(prediction_results["all_predictions"]))):
        pred = prediction_results["all_predictions"][i]
        print(f"    {i+1}. {pred['class_name']} (Class {pred['class_index']}): {pred['confidence']:.6f}")

    print("="*50)

def run_inference(sess, features):
    """Run inference using the extracted features"""
    input_details = sess.get_inputs()
    input_dict = {}
    dtype_mapping = {
        'tensor(float)': np.float32,
        'tensor(int64)': np.int64,
        'tensor(uint8)': np.uint8
    }

    # Prepare input tensor for the model
    for input_detail in input_details:
        if list(features.shape) == list(input_detail.shape):
            input_dict[input_detail.name] = features.astype(dtype_mapping.get(input_detail.type, np.float32))
        else:
            print(f"Shape mismatch: features={features.shape}, model={input_detail.shape}")
            try:
                reshaped_features = np.reshape(features, input_detail.shape)
                input_dict[input_detail.name] = reshaped_features.astype(dtype_mapping.get(input_detail.type, np.float32))
                print(f"  Reshaped features to {input_detail.shape}")
            except Exception as e:
                print(f"  Failed to reshape: {e}")
                sys.exit(1)

    # Run inference
    start_time = time.time()
    outputs = sess.run(None, input_dict)
    inference_time = (time.time() - start_time)*1000  # ms
    print(f"Inference time: {inference_time:.2f} ms, Output shape: {outputs[0].shape}")

    # Process outputs
    result = outputs[0]
    if len(result.shape) == 2:
        result = result[0]

    # Apply softmax and get sorted indices
    result_softmax = softmax(result)
    sorted_indices = np.argsort(result_softmax)[::-1]

    # Generate reports
    prediction_results = generate_prediction_report(result_softmax, sorted_indices)
    benchmark_results = generate_benchmark_report(sess, inference_time)

    return outputs, prediction_results, benchmark_results

def main():
    parser = argparse.ArgumentParser(description='End-to-End Audio Inference Pipeline')
    parser.add_argument('--audio-file', type=str, required=True, help='Path to input WAV audio file')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help=f'Model name (default: {DEFAULT_MODEL})')
    parser.add_argument('--tensor-bits', type=int, default=8, choices=[8, 16], help='Tensor bits for quantization (default: 8)')
    parser.add_argument('--debug-level', type=int, default=0, choices=range(4), help='Debug level (0-3) for TIDL execution (default: 0)')
    parser.add_argument('--detailed-report', action='store_true', help='Generate a detailed performance report')
    args = parser.parse_args()

    # Basic configuration
    WORK_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
    base_artifacts_folder = os.path.join(WORK_DIR, 'model_artifacts', '11_01_06_00', SOC, f'int{args.tensor_bits}')
    models_base_path = os.path.join(WORK_DIR, 'models', 'onnx')

    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found")
        sys.exit(1)

    # Use the default audio config defined earlier
    config = AUDIO_CONFIG_DEFAULT

    # Set environment variables for performance stats
    os.environ["TIDL_RT_PERFSTATS"] = "1"
    if platform.machine() == 'aarch64':
        os.environ["TIDL_RT_DDR_STATS"] = "1"

    # Setup model path
    model_path = os.path.join(models_base_path, f"{args.model}.onnx")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    # Process audio to extract features
    total_start_time = time.time()
    features = preprocess_audio_to_features(args.audio_file, config)

    # Set up inference session
    delegate_options = {
        "artifacts_folder": os.path.join(base_artifacts_folder, f"{args.model}"),
        "debug_level": args.debug_level
    }
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    print(f"\nRunning inference on model: {args.model}")

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
        outputs, prediction_results, benchmark_results = run_inference(sess, features)

        if args.detailed_report:
            generate_detailed_report(
                args, features, outputs, prediction_results, benchmark_results, total_start_time
            )
    except Exception as e:
        print(f"Error during inference: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        if sess:
            del sess

    print(f"\nTotal pipeline time: {time.time() - total_start_time:.2f} seconds")
    print("Audio inference completed successfully")


if __name__ == "__main__":
    main()
