#!/usr/bin/env python3
"""
NVIDIA GPU Monitor

A script to continuously monitor NVIDIA GPU statistics including temperature,
utilization, memory usage, and power consumption. Updates every second with
colored output highlighting significant changes.
"""

import subprocess
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

# Thresholds for highlighting changes
THRESHOLDS = {
    'temperature': 3,       # 3°C change
    'gpu_util': 10,         # 10% change
    'mem_util': 10,         # 10% change
    'mem_used': 500,        # 500MB change
    'power_draw': 10        # 10W change
}

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def run_nvidia_smi() -> Optional[str]:
    """
    Run nvidia-smi command to get GPU stats.
    
    Returns:
        The command output as a string, or None if the command failed
    """
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        return None

def parse_nvidia_smi_output(output: str) -> List[Dict]:
    """
    Parse the output from nvidia-smi into a list of dictionaries.
    
    Args:
        output: String output from nvidia-smi command
        
    Returns:
        List of dictionaries containing parsed GPU data
    """
    gpus = []
    
    for line in output.strip().split('\n'):
        parts = [part.strip() for part in line.split(',')]
        
        # Basic validation of expected fields
        if len(parts) < 8:
            continue
            
        timestamp_str = parts[0]
        gpu_name = parts[1]
        
        try:
            temperature = float(parts[2])
            gpu_util = float(parts[3])
            mem_util = float(parts[4])
            mem_used = float(parts[5])
            mem_total = float(parts[6])
            power_draw = float(parts[7])
            
            gpu_data = {
                'timestamp': timestamp_str,
                'name': gpu_name,
                'temperature': temperature,
                'gpu_util': gpu_util,
                'mem_util': mem_util,
                'mem_used': mem_used,
                'mem_total': mem_total,
                'power_draw': power_draw
            }
            
            gpus.append(gpu_data)
        except (ValueError, IndexError) as e:
            # Skip malformed entries
            continue
            
    return gpus

def format_value_with_highlight(
    metric: str, 
    current: float, 
    previous: Optional[float], 
    unit: str
) -> str:
    """
    Format a numeric value with unit, highlighting if it changed significantly.
    
    Args:
        metric: The name of the metric
        current: Current value
        previous: Previous value (or None if first reading)
        unit: The unit string to append (e.g., "%", "°C")
        
    Returns:
        Formatted string with color highlighting if threshold exceeded
    """
    value_str = f"{current:.1f}{unit}"
    
    # Check if there's a significant change
    if previous is not None:
        threshold = THRESHOLDS.get(metric, 0)
        change = abs(current - previous)
        
        if change >= threshold:
            # Determine color based on whether value increased or decreased
            if current > previous:
                return f"{Colors.BRIGHT_RED}{value_str}{Colors.RESET}"
            else:
                return f"{Colors.BRIGHT_GREEN}{value_str}{Colors.RESET}"
            
    return value_str

def display_gpu_info(gpu_data: Dict, previous_data: Optional[Dict] = None):
    """
    Display formatted GPU information.
    
    Args:
        gpu_data: Dictionary containing current GPU metrics
        previous_data: Dictionary containing previous GPU metrics for comparison
    """
    # Extract key values
    gpu_name = gpu_data['name']
    timestamp = gpu_data['timestamp']
    
    # Convert MB to GB for memory values
    mem_used_gb = gpu_data['mem_used'] / 1024
    mem_total_gb = gpu_data['mem_total'] / 1024
    
    # Get previous values if available
    prev_temp = previous_data['temperature'] if previous_data else None
    prev_gpu_util = previous_data['gpu_util'] if previous_data else None
    prev_mem_util = previous_data['mem_util'] if previous_data else None
    prev_mem_used = previous_data['mem_used'] if previous_data else None
    prev_power = previous_data['power_draw'] if previous_data else None

    # Format current timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Header
    print(f"{Colors.BOLD}{Colors.CYAN}===== NVIDIA GPU Monitor ====={Colors.RESET}")
    print(f"{Colors.BOLD}Time: {Colors.RESET}{current_time}")
    print(f"{Colors.BOLD}GPU:  {Colors.RESET}{gpu_name}")
    print(f"{Colors.BOLD}{'=' * 40}{Colors.RESET}")
    
    # Statistics
    print(f"{Colors.BOLD}Temperature:    {Colors.RESET}{format_value_with_highlight('temperature', gpu_data['temperature'], prev_temp, '°C')}")
    print(f"{Colors.BOLD}GPU Usage:      {Colors.RESET}{format_value_with_highlight('gpu_util', gpu_data['gpu_util'], prev_gpu_util, '%')}")
    print(f"{Colors.BOLD}Memory Usage:   {Colors.RESET}{format_value_with_highlight('mem_util', gpu_data['mem_util'], prev_mem_util, '%')}")
    print(f"{Colors.BOLD}Memory:         {Colors.RESET}{format_value_with_highlight('mem_used', gpu_data['mem_used'], prev_mem_used, 'MB')} / {mem_total_gb:.2f}GB")
    print(f"{Colors.BOLD}Power Draw:     {Colors.RESET}{format_value_with_highlight('power_draw', gpu_data['power_draw'], prev_power, 'W')}")
    
    # Footer
    print(f"{Colors.BOLD}{'=' * 40}{Colors.RESET}")
    print(f"Press Ctrl+C to exit")

def main():
    """Main execution function."""
    previous_data = None
    
    try:
        while True:
            # Check for GPU data
            output = run_nvidia_smi()
            
            if not output:
                clear_screen()
                print(f"{Colors.RED}Error: Could not connect to NVIDIA GPU.{Colors.RESET}")
                print("Make sure the GPU is properly connected and NVIDIA drivers are installed.")
                print("\nRetrying in 3 seconds... (Press Ctrl+C to exit)")
                time.sleep(3)
                continue
                
            # Parse the output
            gpu_list = parse_nvidia_smi_output(output)
            
            if not gpu_list:
                clear_screen()
                print(f"{Colors.YELLOW}Warning: No GPU data available.{Colors.RESET}")
                print("\nRetrying in 3 seconds... (Press Ctrl+C to exit)")
                time.sleep(3)
                continue
            
            # For simplicity, we'll only display the first GPU
            gpu_data = gpu_list[0]
            
            # Display information
            clear_screen()
            display_gpu_info(gpu_data, previous_data)
            
            # Save current data for next comparison
            previous_data = gpu_data
            
            # Wait before next update
            time.sleep(1)
            
    except KeyboardInterrupt:
        clear_screen()
        print(f"{Colors.BRIGHT_CYAN}NVIDIA GPU Monitor stopped.{Colors.RESET}")
        print("Thank you for using the monitor!")
        sys.exit(0)
    except Exception as e:
        clear_screen()
        print(f"{Colors.RED}An unexpected error occurred: {e}{Colors.RESET}")
        sys.exit(1)

if __name__ == "__main__":
    main()

