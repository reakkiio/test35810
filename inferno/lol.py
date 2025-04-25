import os
import requests
import math
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple

def estimate_gguf_ram_requirements(model_path: str, verbose: bool = True) -> Dict[str, float]:
    """
    Estimate RAM requirements to run a GGUF model.
    
    Args:
        model_path: Path to the GGUF model file or URL
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary with RAM requirements for different quantization levels
    """
    # Get model size in bytes
    file_size_bytes = get_model_size(model_path)
    if file_size_bytes is None:
        print(f"Couldn't determine the size of the model at {model_path}")
        return {}
    
    file_size_gb = file_size_bytes / (1024**3)  # Convert to GB
    
    if verbose:
        print(f"Model size: {file_size_gb:.2f} GB")
    
    # Estimate RAM requirements based on model size and quantization
    # These multipliers are based on empirical observations
    ram_requirements = {}
    
    # Comprehensive list of all GGUF quantization levels and their typical RAM multipliers
    # From lowest precision (Q2) to highest (F16/FP16)
    quantization_multipliers = {
        # 2-bit quantization
        "Q2_K": 1.15,       # Q2_K (2-bit quantization with K-quants)
        "Q2_K_S": 1.18,     # Q2_K_S (2-bit quantization with K-quants, small)
        
        # 3-bit quantization
        "Q3_K_S": 1.25,     # Q3_K_S (3-bit quantization with K-quants, small)
        "Q3_K_M": 1.28,     # Q3_K_M (3-bit quantization with K-quants, medium)
        "Q3_K_L": 1.30,     # Q3_K_L (3-bit quantization with K-quants, large)
        
        # 4-bit quantization
        "Q4_0": 1.33,       # Q4_0 (4-bit quantization, version 0)
        "Q4_1": 1.35,       # Q4_1 (4-bit quantization, version 1)
        "Q4_K_S": 1.38,     # Q4_K_S (4-bit quantization with K-quants, small)
        "Q4_K_M": 1.40,     # Q4_K_M (4-bit quantization with K-quants, medium)
        "Q4_K_L": 1.43,     # Q4_K_L (4-bit quantization with K-quants, large)
        
        # 5-bit quantization
        "Q5_0": 1.50,       # Q5_0 (5-bit quantization, version 0)
        "Q5_1": 1.55,       # Q5_1 (5-bit quantization, version 1)
        "Q5_K_S": 1.60,     # Q5_K_S (5-bit quantization with K-quants, small)
        "Q5_K_M": 1.65,     # Q5_K_M (5-bit quantization with K-quants, medium)
        "Q5_K_L": 1.70,     # Q5_K_L (5-bit quantization with K-quants, large)
        
        # 6-bit quantization
        "Q6_K": 1.80,       # Q6_K (6-bit quantization with K-quants)
        
        # 8-bit quantization
        "Q8_0": 2.00,       # Q8_0 (8-bit quantization, version 0)
        "Q8_K": 2.10,       # Q8_K (8-bit quantization with K-quants)
        
        # Floating point formats
        "F16": 2.80,        # F16 (16-bit float, same as FP16)
        "FP16": 2.80,       # FP16 (16-bit float)
    }
    
    # Calculate RAM requirements for each quantization level
    for quant_name, multiplier in quantization_multipliers.items():
        ram_requirements[quant_name] = file_size_gb * multiplier
    
    # For context generation, add additional overhead based on context length
    context_lengths = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    context_ram = {}
    
    # Formula for estimating KV cache size based on model size and context length
    # This formula is approximate and based on empirical observations
    model_params_billions = estimate_params_from_file_size(file_size_bytes, quant="Q4_K_M")
    
    for ctx_len in context_lengths:
        # KV cache formula: 2 (K&V) * num_layers * hidden_dim * context_length * bytes_per_token
        # We estimate based on model parameters
        estimated_layers = min(max(int(model_params_billions * 0.8), 24), 80)  # Estimate number of layers
        estimated_hidden_dim = min(max(int(model_params_billions * 30), 1024), 8192)  # Estimate hidden dimension
        bytes_per_token = 2  # 2 bytes for half-precision (FP16) KV cache
        
        kv_cache_size_gb = (2 * estimated_layers * estimated_hidden_dim * ctx_len * bytes_per_token) / (1024**3)
        context_ram[f"Context {ctx_len}"] = kv_cache_size_gb
    
    ram_requirements["context_overhead"] = context_ram
    
    if verbose:
        print("\nEstimated RAM requirements for running the model:")
        for quant, ram in sorted(
            [(q, r) for q, r in ram_requirements.items() if q != "context_overhead"],
            key=lambda x: x[1]  # Sort by RAM requirement
        ):
            print(f"- {quant}: {ram:.2f} GB base RAM")
        
        print("\nEstimated model parameters: ~{:.1f}B".format(model_params_billions))
        print("\nAdditional RAM for KV cache at different context lengths:")
        for ctx, ram in context_ram.items():
            print(f"- {ctx}: +{ram:.2f} GB")
        
        print("\nTotal RAM examples (sorted by increasing RAM usage):")
        # Show examples for a few representative quantization levels
        example_quants = ["Q2_K", "Q4_K_M", "Q8_0", "F16"]
        for ctx, kv_ram in list(context_ram.items())[:4]:  # Show first 4 context lengths only
            ctx_length = int(ctx.split(" ")[1])
            print(f"\nWith {ctx_length} context length:")
            for quant in example_quants:
                total = ram_requirements[quant] + kv_ram
                print(f"- {quant}: {total:.2f} GB")
    
    return ram_requirements

def estimate_params_from_file_size(file_size_bytes: int, quant: str = "Q4_K_M") -> float:
    """
    Estimate the number of parameters (in billions) from model file size.
    
    Args:
        file_size_bytes: Size of the model file in bytes
        quant: Quantization type
        
    Returns:
        Estimated number of parameters in billions
    """
    # Bits per parameter for different quantization types
    bits_per_param = {
        "Q2_K": 2.5,      # ~2-2.5 bits per param
        "Q3_K_M": 3.5,    # ~3-3.5 bits per param
        "Q4_K_M": 4.5,    # ~4-4.5 bits per param
        "Q5_K_M": 5.5,    # ~5-5.5 bits per param
        "Q6_K": 6.5,      # ~6-6.5 bits per param
        "Q8_0": 8.5,      # ~8-8.5 bits per param
        "F16": 16.0,      # 16 bits per param
    }
    
    # Default to Q4_K_M if the specified quant is not in the dictionary
    bits = bits_per_param.get(quant, 4.5)
    
    # Convert bits to bytes for calculation
    bytes_per_param = bits / 8
    
    # Calculate number of parameters
    params = file_size_bytes / bytes_per_param
    
    # Convert to billions
    params_billions = params / 1e9
    
    return params_billions

def get_model_size(model_path: str) -> Optional[int]:
    """
    Get the size of a model file in bytes.
    Works for both local files and remote URLs.
    
    Args:
        model_path: Path to the model file or URL
        
    Returns:
        Size in bytes or None if size can't be determined
    """
    if os.path.exists(model_path):
        # Local file
        return os.path.getsize(model_path)
    
    elif model_path.startswith(('http://', 'https://')):
        # Remote file - try to get size from HTTP headers
        try:
            response = requests.head(model_path, allow_redirects=True)
            if response.status_code == 200 and 'content-length' in response.headers:
                return int(response.headers['content-length'])
            else:
                print(f"Couldn't get Content-Length header for {model_path}")
                return None
        except Exception as e:
            print(f"Error getting file size from URL: {e}")
            return None
    else:
        print(f"Path {model_path} is neither a valid file nor URL")
        return None

def suggest_hardware(ram_required: float) -> str:
    """
    Suggest hardware based on RAM requirements.
    
    Args:
        ram_required: RAM required in GB
        
    Returns:
        Hardware recommendation
    """
    if ram_required <= 4:
        return "Entry-level desktop/laptop with 8GB RAM should work"
    elif ram_required <= 8:
        return "Standard desktop/laptop with 16GB RAM recommended"
    elif ram_required <= 16:
        return "High-end desktop/laptop with 32GB RAM recommended"
    elif ram_required <= 32:
        return "Workstation with 64GB RAM recommended"
    elif ram_required <= 64:
        return "High-end workstation with 128GB RAM recommended"
    else:
        return f"Server-grade hardware with at least {math.ceil(ram_required*1.5)}GB RAM recommended"

def detect_gpu_vram():
    """
    Detect available GPU VRAM if possible.
    Requires optional dependencies (nvidia-ml-py or pynvml).
    
    Returns:
        Dict mapping GPU index to available VRAM in GB, or empty dict if detection fails
    """
    try:
        import pynvml # type: ignore[import]
        pynvml.nvmlInit()
        
        vram_info = {}
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_total_gb = info.total / (1024**3)
            vram_free_gb = info.free / (1024**3)
            vram_info[i] = {
                "total": vram_total_gb,
                "free": vram_free_gb,
                "name": pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            }
            
        pynvml.nvmlShutdown()
        return vram_info
    
    except ImportError:
        print("GPU VRAM detection requires pynvml. Install with: pip install nvidia-ml-py")
        return {}
    except Exception as e:
        print(f"Error detecting GPUs: {e}")
        return {}


def detect_quantization_from_filename(filename: str) -> Optional[str]:
    """
    Try to detect the quantization type from the filename.
    
    Args:
        filename: Name of the model file
        
    Returns:
        Detected quantization type or None if not detected
    """
    filename = filename.lower()
    
    # Common quantization naming patterns
    patterns = [
        ('q2k', 'Q2_K'),
        ('q2_k', 'Q2_K'),
        ('q3k', 'Q3_K_M'),
        ('q3_k', 'Q3_K_M'),
        ('q3_k_m', 'Q3_K_M'),
        ('q3_k_s', 'Q3_K_S'),
        ('q3_k_l', 'Q3_K_L'),
        ('q4_0', 'Q4_0'),
        ('q4_1', 'Q4_1'),
        ('q4k', 'Q4_K_M'),
        ('q4_k', 'Q4_K_M'),
        ('q4_k_m', 'Q4_K_M'),
        ('q4_k_s', 'Q4_K_S'),
        ('q4_k_l', 'Q4_K_L'),
        ('q5_0', 'Q5_0'),
        ('q5_1', 'Q5_1'),
        ('q5k', 'Q5_K_M'),
        ('q5_k', 'Q5_K_M'),
        ('q5_k_m', 'Q5_K_M'),
        ('q5_k_s', 'Q5_K_S'),
        ('q5_k_l', 'Q5_K_L'),
        ('q6k', 'Q6_K'),
        ('q6_k', 'Q6_K'),
        ('q8_0', 'Q8_0'),
        ('q8k', 'Q8_K'),
        ('q8_k', 'Q8_K'),
        ('f16', 'F16'),
        ('fp16', 'FP16')
    ]
    
    for pattern, quant_type in patterns:
        if pattern in filename:
            return quant_type
    
    return None

def estimate_from_huggingface_repo(repo_id: str, branch: str = "main") -> Dict[str, float]:
    """
    Estimate RAM requirements for a model from a Hugging Face repository.
    
    Args:
        repo_id: Hugging Face repository ID (e.g., 'TheBloke/Llama-2-7B-GGUF')
        branch: Repository branch
        
    Returns:
        Dictionary with RAM requirements
    """
    api_url = f"https://huggingface.co/api/models/{repo_id}/tree/{branch}"
    try:
        response = requests.get(api_url)
        if response.status_code != 200:
            print(f"Error accessing repository: HTTP {response.status_code}")
            return {}
        
        files = response.json()
        gguf_files = [f for f in files if f.get('path', '').endswith('.gguf')]
        
        if not gguf_files:
            print(f"No GGUF files found in repository {repo_id}")
            return {}
        
        print(f"Found {len(gguf_files)} GGUF files in repository")
        
        # Group files by quantization type
        quant_groups = {}
        
        for file in gguf_files:
            file_path = file.get('path', '')
            filename = os.path.basename(file_path)
            size_bytes = file.get('size', 0)
            size_gb = size_bytes / (1024**3)
            
            quant_type = detect_quantization_from_filename(filename)
            if quant_type:
                if quant_type not in quant_groups:
                    quant_groups[quant_type] = []
                quant_groups[quant_type].append((filename, size_gb, size_bytes))
        
        print("\nAvailable quantizations in repository:")
        for quant, files in quant_groups.items():
            for filename, size_gb, _ in files:
                print(f"- {quant}: {filename} ({size_gb:.2f} GB)")
        
        # Find a representative file for RAM estimation
        # Prefer Q4_K_M as it's common, or pick the largest file
        if not quant_groups:
            # If quantization detection failed, just use the largest file
            largest_file = max(gguf_files, key=lambda x: x.get('size', 0))
            size_bytes = largest_file.get('size', 0)
            file_path = largest_file.get('path', '')
            print(f"\nUsing largest GGUF model for estimation: {file_path} ({size_bytes / (1024**3):.2f} GB)")
            return estimate_gguf_ram_requirements(file_path, verbose=False)
        
        # Choose a representative model
        chosen_quant = None
        chosen_file = None
        
        # Preference order
        preferred_quants = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]
        
        for quant in preferred_quants:
            if quant in quant_groups:
                chosen_quant = quant
                # Choose the latest version if multiple files with same quant
                chosen_file = max(quant_groups[quant], key=lambda x: x[1])  # Sort by size
                break
        
        if not chosen_quant:
            # Just choose the first available quantization
            chosen_quant = list(quant_groups.keys())[0]
            chosen_file = quant_groups[chosen_quant][0]
        
        filename, size_gb, size_bytes = chosen_file
        print(f"\nUsing {chosen_quant} model for estimation: {filename} ({size_gb:.2f} GB)")
        
        # Create RAM estimation using the file size
        ram_requirements = {}
        
        # Use the same estimation logic as the main function
        # Comprehensive list of all GGUF quantization levels
        quantization_multipliers = {
            # 2-bit quantization
            "Q2_K": 1.15,     
            "Q2_K_S": 1.18,   
            
            # 3-bit quantization
            "Q3_K_S": 1.25,   
            "Q3_K_M": 1.28,   
            "Q3_K_L": 1.30,   
            
            # 4-bit quantization
            "Q4_0": 1.33,     
            "Q4_1": 1.35,     
            "Q4_K_S": 1.38,   
            "Q4_K_M": 1.40,   
            "Q4_K_L": 1.43,   
            
            # 5-bit quantization
            "Q5_0": 1.50,     
            "Q5_1": 1.55,     
            "Q5_K_S": 1.60,   
            "Q5_K_M": 1.65,   
            "Q5_K_L": 1.70,   
            
            # 6-bit quantization
            "Q6_K": 1.80,     
            
            # 8-bit quantization
            "Q8_0": 2.00,     
            "Q8_K": 2.10,     
            
            # Floating point formats
            "F16": 2.80,      
            "FP16": 2.80,     
        }
        
        # Estimate base size from the chosen quantization
        base_size_gb = size_bytes / (1024**3)
        model_params_billions = estimate_params_from_file_size(size_bytes, chosen_quant)
        
        # Calculate RAM estimates for all quantizations by scaling from the chosen one
        chosen_multiplier = quantization_multipliers[chosen_quant]
        base_model_size = base_size_gb / chosen_multiplier  # Theoretical unquantized size
        
        for quant_name, multiplier in quantization_multipliers.items():
            ram_requirements[quant_name] = base_model_size * multiplier
        
        # For context generation, add additional overhead
        context_lengths = [2048, 4096, 8192, 16384, 32768, 65536]
        context_ram = {}
        
        # KV cache formula
        estimated_layers = min(max(int(model_params_billions * 0.8), 24), 80)
        estimated_hidden_dim = min(max(int(model_params_billions * 30), 1024), 8192)
        bytes_per_token = 2  # 2 bytes for half-precision KV cache
        
        for ctx_len in context_lengths:
            kv_cache_size_gb = (2 * estimated_layers * estimated_hidden_dim * ctx_len * bytes_per_token) / (1024**3)
            context_ram[f"Context {ctx_len}"] = kv_cache_size_gb
        
        ram_requirements["context_overhead"] = context_ram
        ram_requirements["model_params_billions"] = model_params_billions
        
        return ram_requirements
    
    except Exception as e:
        print(f"Error accessing Hugging Face repository: {e}")
        return {}

def print_gpu_compatibility(ram_requirements: Dict[str, float], vram_info: Dict):
    """
    Print GPU compatibility information based on RAM requirements.
    
    Args:
        ram_requirements: Dictionary with RAM requirements
        vram_info: Dictionary with GPU VRAM information
    """
    if not vram_info:
        print("\nNo GPU information available")
        return
    
    print("\n=== GPU Compatibility Analysis ===")
    
    # Context lengths to analyze
    context_lengths = [2048, 4096, 8192, 16384]
    
    # Quantization levels to check (arranged from most efficient to highest quality)
    quant_levels = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]
    
    # Get context RAM overhead
    context_ram = ram_requirements.get("context_overhead", {})
    
    # For each GPU
    for gpu_idx, gpu_data in vram_info.items():
        gpu_name = gpu_data.get("name", f"GPU {gpu_idx}")
        vram_total = gpu_data.get("total", 0)
        vram_free = gpu_data.get("free", 0)
        
        print(f"\n{gpu_name}: {vram_total:.2f} GB total VRAM, {vram_free:.2f} GB free")
        
        # Check compatibility for each combination
        print("\nCompatibility matrix (✓: fits, ✗: doesn't fit):")
        
        # Print header row with context lengths
        header = "Quantization | "
        for ctx_len in context_lengths:
            header += f"{ctx_len:6d} | "
        print(header)
        print("-" * len(header))
        
        # Print compatibility for each quantization level
        for quant in quant_levels:
            if quant not in ram_requirements:
                continue
                
            base_ram = ram_requirements[quant]
            row = f"{quant:11s} | "
            
            for ctx_len in context_lengths:
                ctx_key = f"Context {ctx_len}"
                if ctx_key in context_ram:
                    ctx_overhead = context_ram[ctx_key]
                    total_ram = base_ram + ctx_overhead
                    
                    # Check if it fits in VRAM
                    fits = total_ram <= vram_free
                    row += f"{'✓':6s} | " if fits else f"{'✗':6s} | "
                else:
                    row += f"{'?':6s} | "
            
            print(row)
    
    print("\nRecommendations:")
    
    # Find best quantization/context combination that fits
    best_quant = None
    best_ctx = None
    
    # Start with highest quality and largest context
    for quant in reversed(quant_levels):
        if quant not in ram_requirements:
            continue
            
        base_ram = ram_requirements[quant]
        
        for ctx_len in reversed(context_lengths):
            ctx_key = f"Context {ctx_len}"
            if ctx_key in context_ram:
                ctx_overhead = context_ram[ctx_key]
                total_ram = base_ram + ctx_overhead
                
                # Check if any GPU can run this configuration
                for _, gpu_data in vram_info.items():
                    vram_free = gpu_data.get("free", 0)
                    if total_ram <= vram_free:
                        best_quant = quant
                        best_ctx = ctx_len
                        break
                
                if best_quant:
                    break
        
        if best_quant:
            break
    
    if best_quant:
        print(f"- Recommended configuration: {best_quant} with context length {best_ctx}")
    else:
        print("- Your GPU(s) may not have enough VRAM to run this model efficiently.")
        print("- Consider using a smaller model or a lower quantization level.")

# Example usage
if __name__ == "__main__":
    # Example 1: Estimate from local file
    print("==== Local GGUF Model Example ====")
    model_path = "path/to/your/model.gguf"  # Change this to your model path
    
    if os.path.exists(model_path):
        ram_reqs = estimate_gguf_ram_requirements(model_path)
        if "Q4_K_M" in ram_reqs:
            q4_ram = ram_reqs["Q4_K_M"]
            print(f"\nHardware suggestion: {suggest_hardware(q4_ram)}")
    else:
        print(f"Model file {model_path} not found. Skipping local example.")
    
    # Example 2: Estimate from Hugging Face repository
    print("\n==== Hugging Face Model Example ====")
    repo_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"  # Example repository
    ram_reqs = estimate_from_huggingface_repo(repo_id)
    
    if ram_reqs:
        print("\nEstimated RAM requirements:")
        for quant in ["Q2_K", "Q4_K_M", "Q8_0", "F16"]:
            if quant in ram_reqs:
                ram = ram_reqs[quant]
                print(f"- {quant}: {ram:.2f} GB")
        
        print(f"\nEstimated parameters: ~{ram_reqs.get('model_params_billions', 0):.1f}B")
        
        if "Q4_K_M" in ram_reqs:
            q4_ram = ram_reqs["Q4_K_M"]
            print(f"\nHardware suggestion: {suggest_hardware(q4_ram)}")
    
    # Example 3: Check GPU VRAM and compatibility
    print("\n==== GPU VRAM Detection ====")
    vram_info = detect_gpu_vram()
    
    if ram_reqs and vram_info:
        print_gpu_compatibility(ram_reqs, vram_info)
