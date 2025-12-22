#!/bin/bash
# filepath: download_models.sh

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source VERSION file for TIDL_VER and SOC
if [ ! -f "$SCRIPT_DIR/VERSION" ]; then
    echo "Error: VERSION file not found at $SCRIPT_DIR/VERSION"
    exit 1
fi
source "$SCRIPT_DIR/VERSION"

# Validate required variables
if [ -z "${TIDL_VER}" ]; then
    echo "Error: TIDL_VER is not set in VERSION file"
    exit 1
fi

# Use SOC from environment, default to am62a if not set
if [ -z "${SOC}" ]; then
    SOC=am62a
fi

# Model files to download
MODELS=(
    "gtcrn_dns3.onnx"
    "vggish11_20250324-1807_ptq.onnx"
    "yamnet_combined.onnx"
)

# Configuration
BASE_URL="https://software-dl.ti.com/jacinto7/esd/modelzoo/audioai/models/onnx"
LOCAL_MODELS_DIR="$SCRIPT_DIR/models/onnx"

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

# Function to list download URLs
list_urls() {
    for model in "${MODELS[@]}"; do
        echo "$BASE_URL/$model"
    done
}

# Function to download selected models
download_models() {
    local models=("$@")
    local success_count=0
    local fail_count=0
    
    echo
    print_color $CYAN "Starting download of ${#models[@]} model(s)..."
    echo
    
    ensure_dir "$LOCAL_MODELS_DIR"
    
    for model in "${models[@]}"; do
        local local_path="$LOCAL_MODELS_DIR/$model"
        local remote_url="$BASE_URL/$model"
        
        print_color $YELLOW "Downloading: $model"
        print_color $CYAN "  From: $remote_url"
        
        if wget -O "$local_path" "$remote_url"; then
            if [ -s "$local_path" ]; then
                print_color $GREEN "✓ Downloaded: $model"
                ((success_count++))
            else
                print_color $RED "✗ Downloaded file is empty: $model"
                rm -f "$local_path"
                ((fail_count++))
            fi
        else
            print_color $RED "✗ Failed to download: $model"
            rm -f "$local_path"
            ((fail_count++))
        fi
        echo
    done
    
    print_color $GREEN "Download process completed!"
    print_color $CYAN "Models saved to: $LOCAL_MODELS_DIR"
    print_color $CYAN "Summary: $success_count succeeded, $fail_count failed"
}

# Main execution
main() {
    print_color $BLUE "AudioAI ModelZoo Downloader"
    print_color $BLUE "============================"
    echo
    
    # Check for wget
    if ! command -v wget &> /dev/null; then
        print_color $RED "Error: wget is not available. Please install wget."
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
        -l|--list-urls)
            list_urls
            exit 0
            ;;
        -h|--help)
            print_color $BLUE "AudioAI ModelZoo Downloader"
            print_color $BLUE "==========================="
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
            print_color $CYAN "  --list-urls          Print download URLs for all models"
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
