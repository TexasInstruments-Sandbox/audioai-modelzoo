#!/usr/bin/env python3
"""
VGGish Audio Classification Inference Script

End-to-end audio inference pipeline for VGGish on TI devices using TIDL.
This script processes an input audio file and runs inference using ONNX Runtime
with TIDLExecutionProvider for hardware acceleration.

Usage:
    python3 vggish_infer_audio.py --audio-file sample_wav/dog_bark.wav
    python3 vggish_infer_audio.py --audio-file sample_wav/street_music.wav
"""

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from audio_processing import preprocess_audio_waveform, log_mel_spectrogram

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

# Audio processing configuration for VGGish
AUDIO_CONFIG_DEFAULT = {
    'dataset': {
        'sample_rate': 16000,
        'duration': 4.0,
        'n_fft': 1024,
        'hop_length': 512,
        'n_mels': 64
    }
}

# UrbanSound8K class label mapping (10 classes)
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

# Environment setup
if not (SOC := os.environ.get("SOC")):
    sys.exit("Error: SOC environment variable is not defined")

# Default model for inferenceython3 vggish_infer_audio.py
DEFAULT_MODEL = 'vggish11_20250324-1807_ptq'

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

def preprocess_audio_to_features(audio_path, config):
    """Process audio file to extract features suitable for model input.
    
    Returns:
        tuple: (mel_spectrogram_features, audio_duration)
    """
    import soundfile as sf
    
    print(f"\nProcessing audio file: {audio_path}")

    # Get original audio duration before preprocessing
    audio_info = sf.info(audio_path)
    original_duration = audio_info.duration

    # Preprocess audio waveform and extract log mel spectrogram features
    start_time = time.time()
    waveform = preprocess_audio_waveform(
        audio_path,
        sample_rate=config['dataset']['sample_rate'],
        duration=config['dataset']['duration']
    )
    mel_spec = log_mel_spectrogram(waveform, config)
    preprocessing_time_ms = (time.time() - start_time) * 1000
    
    print(f"  Waveform shape: {waveform.shape}")
    print(f"  Log mel spectrogram shape: {mel_spec.shape}")
    print(f"  Total preprocessing time: {preprocessing_time_ms:.2f} ms")

    return mel_spec.cpu().numpy(), original_duration

def format_class_name(idx):
    """Format class index with name from built-in class labels."""
    str_idx = str(idx)
    class_name = CLASS_LABELS.get(str_idx)
    return f"Class {idx} ({class_name})" if class_name else f"Class {idx}"

def softmax(x):
    """Apply softmax function to input array."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

def generate_prediction_report(result_softmax, sorted_indices):
    """Generate and display prediction report."""
    top_indices = sorted_indices[:5]
    prediction_results = {"top_predictions": [], "all_predictions": []}

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
    """Generate and display benchmark report."""
    benchmark_dict = sess.get_TI_benchmark_data()
    total_time, subgraph_proc_time, copy_time, ddr_read, ddr_write = get_benchmark_output(benchmark_dict)

    # Store and display benchmark results
    benchmark_results = {
        "total_time_ms": float(total_time),
        "processing_time_ms": float(subgraph_proc_time),
        "copy_time_ms": float(copy_time),
        "ddr_read_MBs": float(ddr_read),
        "ddr_write_MBs": float(ddr_write),
        "inference_time_ms": float(inference_time)
    }

    print("\nTIDL Performance:")
    print(f"  Total time: {total_time:.2f} ms")
    print(f"  Processing time: {subgraph_proc_time:.2f} ms")
    print(f"  Copy time: {copy_time:.2f} ms")
    print(f"  Inference time: {inference_time:.2f} ms")
    if ddr_read > 0 or ddr_write > 0:
        print(f"  DDR read: {ddr_read:.2f} MB/s, DDR write: {ddr_write:.2f} MB/s")

    return benchmark_results

def generate_summary_report(prediction_results, benchmark_results, audio_duration, 
                           model_name=None, tensor_bits=None, audio_filename=None):
    """Generate standardized summary report for both CLI and notebook."""
    separator = "=" * 70
    print(f"\n{separator}")
    print("VGGISH11 INFERENCE SUMMARY")
    print(separator)
    
    if audio_filename:
        print(f"\nAudio File: {audio_filename}")
    if model_name and tensor_bits:
        print(f"Model: {model_name} (int{tensor_bits})")
    print(f"Audio Duration: {audio_duration:.2f} seconds")
    
    # Top prediction
    if prediction_results and len(prediction_results['top_predictions']) > 0:
        top_pred = prediction_results['top_predictions'][0]
        print(f"Top Class: {top_pred['class_name']} (Class {top_pred['class_index']}) - Confidence: {top_pred['confidence']:.6f}")
    
    # Performance metrics
    process_time_ms = max(
        benchmark_results['inference_time_ms'], 
        benchmark_results['processing_time_ms']
    )
    rtf = process_time_ms / 1000 / audio_duration
    
    print(f"\nPerformance:")
    print(f"  Inference: {benchmark_results['inference_time_ms']:.2f} ms (TIDL: {benchmark_results['processing_time_ms']:.2f} ms, Copy: {benchmark_results['copy_time_ms']:.2f} ms)")
    print(f"  Throughput: RTF = {rtf:.4f} ({process_time_ms/1000:.3f}s / {audio_duration:.2f}s)")
    if benchmark_results['ddr_read_MBs'] > 0 or benchmark_results['ddr_write_MBs'] > 0:
        print(f"  Memory: {benchmark_results['ddr_read_MBs']:.2f} MB/s read | {benchmark_results['ddr_write_MBs']:.2f} MB/s write")
    print(f"{separator}")

def generate_detailed_report(args, features, outputs, prediction_results, benchmark_results, 
                            total_start_time, audio_duration):
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
    rtf = process_time_ms / 1000 / audio_duration

    separator = "=" * 50
    print(f"\n{separator}")
    print("DETAILED PERFORMANCE REPORT")
    print(separator)
    print(f"Model: {args.model} (int{args.tensor_bits})")
    print(f"Audio file: {os.path.basename(args.audio_file)}")
    print(f"Feature shape: {features.shape}, Output shape: {outputs[0].shape}")

    total_pipeline_time_ms = total_pipeline_time * 1000
    inference_time_ms = benchmark_results['inference_time_ms']
    preprocessing_time_ms = total_pipeline_time_ms - inference_time_ms
    
    print("\nPipeline Time Breakdown:")
    print(f"  Total pipeline time: {total_pipeline_time_ms:.2f} ms")
    print(f"  Preprocessing time: {preprocessing_time_ms:.2f} ms")
    print(f"  Inference time: {inference_time_ms:.2f} ms")
    print(f"  Audio duration: {audio_duration:.2f} seconds")
    print(f"  Real-time factor (RTF): {rtf:.4f} ({process_time_ms/1000:.3f}s / {audio_duration:.2f}s)")

    print("\nTop Recognized Class:")
    if len(prediction_results["all_predictions"]) > 0:
        pred = prediction_results["all_predictions"][0]
        print(f"  {pred['class_name']} (Class {pred['class_index']}): {pred['confidence']:.6f}")

    print(separator)

def run_inference(sess, features):
    """Run inference using the extracted features."""
    DTYPE_MAPPING = {
        'tensor(float)': np.float32,
        'tensor(int64)': np.int64,
        'tensor(uint8)': np.uint8
    }
    
    input_details = sess.get_inputs()
    input_dict = {}

    # Prepare input tensor for the model
    for input_detail in input_details:
        target_dtype = DTYPE_MAPPING.get(input_detail.type, np.float32)
        
        if list(features.shape) == list(input_detail.shape):
            input_dict[input_detail.name] = features.astype(target_dtype)
        else:
            print(f"Shape mismatch: features={features.shape}, model={input_detail.shape}")
            try:
                reshaped_features = np.reshape(features, input_detail.shape)
                input_dict[input_detail.name] = reshaped_features.astype(target_dtype)
                print(f"  Reshaped features to {input_detail.shape}")
            except Exception as e:
                sys.exit(f"  Failed to reshape: {e}")

    # Run inference
    start_time = time.time()
    outputs = sess.run(None, input_dict)
    inference_time_ms = (time.time() - start_time) * 1000
    print(f"Inference time: {inference_time_ms:.2f} ms, Output shape: {outputs[0].shape}")

    # Process outputs
    result = outputs[0]
    if len(result.shape) == 2:
        result = result[0]

    # Apply softmax and get sorted indices
    result_softmax = softmax(result)
    sorted_indices = np.argsort(result_softmax)[::-1]

    # Generate reports
    prediction_results = generate_prediction_report(result_softmax, sorted_indices)
    benchmark_results = generate_benchmark_report(sess, inference_time_ms)

    return outputs, prediction_results, benchmark_results

def main():
    """Main entry point for VGGish audio inference pipeline."""
    parser = argparse.ArgumentParser(description='VGGish11 Audio Inference Pipeline')
    parser.add_argument('--audio-file', type=str, default='sample_wav/139951-9-0-9.wav', help='Path to input WAV audio file')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help=f'Model name (default: {DEFAULT_MODEL})')
    parser.add_argument('--tensor-bits', type=int, default=8, choices=[8, 16], help='Tensor bits for quantization (default: 8)')
    parser.add_argument('--debug-level', type=int, default=0, choices=range(4), help='Debug level (0-3) for TIDL execution (default: 0)')
    args = parser.parse_args()

    # Set up paths
    work_dir = Path(__file__).parent.parent.parent.resolve()
    base_artifacts_folder = work_dir / 'model_artifacts' / TIDL_VER / SOC
    models_base_path = work_dir / 'models' / 'onnx'

    # Validate audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        sys.exit(f"Error: Audio file '{args.audio_file}' not found")

    # Use the default audio config
    config = AUDIO_CONFIG_DEFAULT

    # Set environment variables for performance stats
    os.environ["TIDL_RT_PERFSTATS"] = "1"
    if platform.machine() == 'aarch64':
        os.environ["TIDL_RT_DDR_STATS"] = "1"

    # Validate model exists
    model_path = models_base_path / f"{args.model}.onnx"
    if not model_path.exists():
        sys.exit(f"Error: Model file not found: {model_path}")

    # Process audio to extract features
    total_start_time = time.time()
    features, audio_duration = preprocess_audio_to_features(args.audio_file, config)

    # Set up inference session
    artifacts_folder = base_artifacts_folder / f"{args.model}_int{args.tensor_bits}"
    delegate_options = {
        "artifacts_folder": str(artifacts_folder),
        "debug_level": args.debug_level
    }
    session_options = ort.SessionOptions()
    session_options.log_severity_level = 3
    
    print(f"\nRunning inference on model: {args.model}")

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
        outputs, prediction_results, benchmark_results = run_inference(sess, features)

        # Generate summary report
        generate_summary_report(
            prediction_results,
            benchmark_results,
            audio_duration,
            model_name=args.model,
            tensor_bits=args.tensor_bits,
            audio_filename=os.path.basename(args.audio_file)
        )
    except Exception as e:
        sys.exit(f"Error during inference: {e}")
    finally:
        if sess:
            del sess

    print("\nVGGish audio inference completed successfully")


if __name__ == "__main__":
    main()
