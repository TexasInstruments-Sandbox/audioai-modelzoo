#!/bin/bash
# filepath: download_models.sh

set -e

# Configuration
TIDL_VER="11_01_06_00"
SOC="am62a"

# Model files to download
MODELS=(
    "gtcrn_dns3.onnx"
    "vggish11_20250324-1807_ptq.onnx"
    "yamnet_combined.onnx"
)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
BASE_URL="https://software-dl.ti.com/jacinto7/esd/modelzoo/audioai/${TIDL_VER}/models/onnx"
LOCAL_MODELS_DIR="$SCRIPT_DIR/models/onnx"  # Download models to models/onnx subdirectory

# Colors for better UI
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    echo -e "${1}${2}${NC}"
}

# Function to create directory if it doesn't exist
ensure_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        print_color $GREEN "Created directory: $1"
    fi
}

# Function to display models menu
display_models_menu() {
    local selected=()
    
    # Initialize all models as selected
    for ((i=0; i<${#MODELS[@]}; i++)); do
        selected[i]=1
    done
    
    while true; do
        clear
        print_color $BLUE "╔══════════════════════════════════════════════════════════════════╗"
        print_color $BLUE "║                    AudioAI ModelZoo Downloader                   ║"
        print_color $BLUE "╚══════════════════════════════════════════════════════════════════╝"
        echo
        print_color $CYAN "TIDL Version: $TIDL_VER | SoC: $SOC"
        echo
        print_color $CYAN "Available Models (✓ = selected, ✗ = deselected):"
        echo
        
        for ((i=0; i<${#MODELS[@]}; i++)); do
            local model="${MODELS[i]}"
            local status="✗"
            local color=$RED
            
            if [ "${selected[i]}" -eq 1 ]; then
                status="✓"
                color=$GREEN
            fi
            
            printf "${color}[%2d] %s %s${NC}\n" $((i+1)) "$status" "$model"
        done
        
        echo
        print_color $YELLOW "Commands:"
        print_color $YELLOW "  [number]     - Toggle selection for model"
        print_color $YELLOW "  a            - Select all models"
        print_color $YELLOW "  n            - Deselect all models"
        print_color $YELLOW "  d            - Download selected models"
        print_color $YELLOW "  q            - Quit"
        echo
        read -p "Enter your choice: " choice
        
        case "$choice" in
            [1-9]|[1-9][0-9]|[1-9][0-9][0-9])
                local idx=$((choice-1))
                if [ $idx -ge 0 ] && [ $idx -lt ${#MODELS[@]} ]; then
                    if [ "${selected[idx]}" -eq 1 ]; then
                        selected[idx]=0
                    else
                        selected[idx]=1
                    fi
                fi
                ;;
            a|A)
                for ((i=0; i<${#MODELS[@]}; i++)); do
                    selected[i]=1
                done
                ;;
            n|N)
                for ((i=0; i<${#MODELS[@]}; i++)); do
                    selected[i]=0
                done
                ;;
            d|D)
                local selected_models=()
                for ((i=0; i<${#MODELS[@]}; i++)); do
                    if [ "${selected[i]}" -eq 1 ]; then
                        selected_models+=("${MODELS[i]}")
                    fi
                done
                
                if [ ${#selected_models[@]} -eq 0 ]; then
                    print_color $RED "No models selected for download!"
                    read -p "Press Enter to continue..."
                else
                    download_models "${selected_models[@]}"
                    return
                fi
                ;;
            q|Q)
                print_color $YELLOW "Goodbye!"
                exit 0
                ;;
            *)
                print_color $RED "Invalid choice. Please try again."
                sleep 1
                ;;
        esac
    done
}

# Function to download selected models
download_models() {
    local models=("$@")
    
    echo
    print_color $CYAN "Starting download of ${#models[@]} model(s)..."
    echo
    
    # Create the local models directory if it doesn't exist
    ensure_dir "$LOCAL_MODELS_DIR"
    
    for model in "${models[@]}"; do
        local local_path="$LOCAL_MODELS_DIR/$model"
        local remote_url="$BASE_URL/$model"
        
        print_color $YELLOW "Downloading: $model"
        print_color $CYAN "  From: $remote_url"
        
        # Download the file
        if command -v wget &> /dev/null; then
            if wget --progress=bar:force -O "$local_path" "$remote_url" 2>&1; then
                print_color $GREEN "✓ Downloaded: $model"
            else
                print_color $RED "✗ Failed to download: $model"
            fi
        elif command -v curl &> /dev/null; then
            if curl -L --progress-bar -o "$local_path" "$remote_url"; then
                print_color $GREEN "✓ Downloaded: $model"
            else
                print_color $RED "✗ Failed to download: $model"
            fi
        fi
        echo
    done
    
    print_color $GREEN "Download process completed!"
    print_color $CYAN "Models saved to: $LOCAL_MODELS_DIR"
}

# Main execution
main() {
    print_color $BLUE "AudioAI ModelZoo Downloader"
    print_color $BLUE "=========================="
    echo
    
    # Check for required tools
    if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
        print_color $RED "Error: Neither wget nor curl is available. Please install one of them."
        exit 1
    fi
    
    # Non-interactive mode: download all models
    if [ "$NON_INTERACTIVE" = true ]; then
        print_color $CYAN "Non-interactive mode: downloading all ${#MODELS[@]} model(s)..."
        print_color $CYAN "TIDL Version: $TIDL_VER | SoC: $SOC"
        echo
        for model in "${MODELS[@]}"; do
            print_color $GREEN "  → $model"
        done
        echo
        download_models "${MODELS[@]}"
        return
    fi
    
    # Interactive mode
    display_models_menu
}

# Parse command line arguments
NON_INTERACTIVE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes)
            NON_INTERACTIVE=true
            shift
            ;;
        -h|--help)
            print_color $BLUE "AudioAI ModelZoo Downloader"
            print_color $BLUE "============================"
            echo
            print_color $YELLOW "Usage: $0 [OPTIONS]"
            print_color $YELLOW "       $0 -h|--help"
            echo
            print_color $CYAN "Configuration:"
            print_color $CYAN "  TIDL Version: $TIDL_VER"
            print_color $CYAN "  SoC: $SOC"
            echo
            print_color $CYAN "Options:"
            print_color $CYAN "  -y, --yes            Non-interactive mode, download all models"
            print_color $CYAN "  -h, --help           Show this help message"
            echo
            print_color $CYAN "Examples:"
            print_color $CYAN "  $0                   # Interactive mode"
            print_color $CYAN "  $0 -y                # Non-interactive, download all models"
            echo
            print_color $CYAN "Available models:"
            for model in "${MODELS[@]}"; do
                print_color $CYAN "  - $model"
            done
            exit 0
            ;;
        *)
            print_color $RED "Unknown option: $1"
            print_color $YELLOW "Use '$0 -h' for help"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"
