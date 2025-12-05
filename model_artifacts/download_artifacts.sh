#!/bin/bash
# filepath: model_artifacts/download_artifacts.sh

set -e

# Configuration
TIDL_VER="11_01_06_00"
SOC="am62a"

# Artifact files to download
ARTIFACTS=(
    "vggish11_20250324-1807_ptq_int8.tar.gz"
    "yamnet_combined_int8.tar.gz"
)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
BASE_URL="https://software-dl.ti.com/jacinto7/esd/modelzoo/audioai/${TIDL_VER}/modelartifacts/${SOC}"
LOCAL_ARTIFACTS_DIR="$SCRIPT_DIR/${TIDL_VER}/${SOC}"  # Download to versioned subdirectory

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

# Function to extract archive
extract_archive() {
    local file="$1"
    local dir="$2"
    
    print_color $CYAN "Extracting: $(basename "$file")"
    
    case "$file" in
        *.tar.gz|*.tgz)
            if tar -xzf "$file" -C "$dir" 2>/dev/null; then
                print_color $GREEN "✓ Extracted: $(basename "$file")"
                rm "$file" 2>/dev/null
                print_color $YELLOW "Removed archive: $(basename "$file")"
                return 0
            else
                print_color $RED "✗ Failed to extract: $(basename "$file")"
                return 1
            fi
            ;;
        *.zip)
            if unzip -q "$file" -d "$dir" 2>/dev/null; then
                print_color $GREEN "✓ Extracted: $(basename "$file")"
                rm "$file" 2>/dev/null
                print_color $YELLOW "Removed archive: $(basename "$file")"
                return 0
            else
                print_color $RED "✗ Failed to extract: $(basename "$file")"
                return 1
            fi
            ;;
        *.tar)
            if tar -xf "$file" -C "$dir" 2>/dev/null; then
                print_color $GREEN "✓ Extracted: $(basename "$file")"
                rm "$file" 2>/dev/null
                print_color $YELLOW "Removed archive: $(basename "$file")"
                return 0
            else
                print_color $RED "✗ Failed to extract: $(basename "$file")"
                return 1
            fi
            ;;
        *)
            print_color $YELLOW "No extraction needed for: $(basename "$file")"
            return 0
            ;;
    esac
}

# Function to display artifacts menu
display_artifacts_menu() {
    local selected=()
    
    # Initialize all artifacts as selected
    for ((i=0; i<${#ARTIFACTS[@]}; i++)); do
        selected[i]=1
    done
    
    while true; do
        clear
        print_color $BLUE "╔══════════════════════════════════════════════════════════════════╗"
        print_color $BLUE "║                AudioAI Artifacts Downloader                      ║"
        print_color $BLUE "╚══════════════════════════════════════════════════════════════════╝"
        echo
        print_color $CYAN "TIDL Version: $TIDL_VER | SoC: $SOC"
        echo
        print_color $CYAN "Available Artifacts (✓ = selected, ✗ = deselected):"
        echo
        
        for ((i=0; i<${#ARTIFACTS[@]}; i++)); do
            local artifact="${ARTIFACTS[i]}"
            local status="✗"
            local color=$RED
            
            if [ "${selected[i]}" -eq 1 ]; then
                status="✓"
                color=$GREEN
            fi
            
            printf "${color}[%2d] %s %s${NC}\n" $((i+1)) "$status" "$artifact"
        done
        
        echo
        print_color $YELLOW "Commands:"
        print_color $YELLOW "  [number]     - Toggle selection for artifact"
        print_color $YELLOW "  a            - Select all artifacts"
        print_color $YELLOW "  n            - Deselect all artifacts"
        print_color $YELLOW "  d            - Download selected artifacts"
        print_color $YELLOW "  q            - Quit"
        echo
        read -p "Enter your choice: " choice
        
        case "$choice" in
            [1-9]|[1-9][0-9]|[1-9][0-9][0-9])
                local idx=$((choice-1))
                if [ $idx -ge 0 ] && [ $idx -lt ${#ARTIFACTS[@]} ]; then
                    if [ "${selected[idx]}" -eq 1 ]; then
                        selected[idx]=0
                    else
                        selected[idx]=1
                    fi
                fi
                ;;
            a|A)
                for ((i=0; i<${#ARTIFACTS[@]}; i++)); do
                    selected[i]=1
                done
                ;;
            n|N)
                for ((i=0; i<${#ARTIFACTS[@]}; i++)); do
                    selected[i]=0
                done
                ;;
            d|D)
                local selected_artifacts=()
                for ((i=0; i<${#ARTIFACTS[@]}; i++)); do
                    if [ "${selected[i]}" -eq 1 ]; then
                        selected_artifacts+=("${ARTIFACTS[i]}")
                    fi
                done
                
                if [ ${#selected_artifacts[@]} -eq 0 ]; then
                    print_color $RED "No artifacts selected for download!"
                    read -p "Press Enter to continue..."
                else
                    download_artifacts "${selected_artifacts[@]}"
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

# Function to download selected artifacts
download_artifacts() {
    local artifacts=("$@")
    
    echo
    print_color $CYAN "Starting download of ${#artifacts[@]} artifact(s)..."
    echo
    
    # Create the local artifacts directory if it doesn't exist
    ensure_dir "$LOCAL_ARTIFACTS_DIR"
    
    for artifact in "${artifacts[@]}"; do
        local local_path="$LOCAL_ARTIFACTS_DIR/$artifact"
        local remote_url="$BASE_URL/$artifact"
        
        print_color $YELLOW "Downloading: $artifact"
        print_color $CYAN "  From: $remote_url"
        
        # Download the file
        local download_success=0
        if command -v wget &> /dev/null; then
            if wget --progress=bar:force -O "$local_path" "$remote_url" 2>&1; then
                download_success=1
                print_color $GREEN "✓ Downloaded: $artifact"
            else
                print_color $RED "✗ Failed to download: $artifact"
            fi
        elif command -v curl &> /dev/null; then
            if curl -L --progress-bar -o "$local_path" "$remote_url"; then
                download_success=1
                print_color $GREEN "✓ Downloaded: $artifact"
            else
                print_color $RED "✗ Failed to download: $artifact"
            fi
        fi
        
        # Extract if download was successful
        if [ $download_success -eq 1 ]; then
            extract_archive "$local_path" "$LOCAL_ARTIFACTS_DIR"
        fi
        
        echo
    done
    
    print_color $GREEN "Download process completed!"
    print_color $CYAN "Artifacts saved to: $LOCAL_ARTIFACTS_DIR"
}

# Main execution
main() {
    print_color $BLUE "AudioAI Artifacts Downloader"
    print_color $BLUE "============================"
    echo
    
    # Check for required tools
    if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
        print_color $RED "Error: Neither wget nor curl is available. Please install one of them."
        exit 1
    fi
    
    # Non-interactive mode: download all artifacts
    if [ "$NON_INTERACTIVE" = true ]; then
        print_color $CYAN "Non-interactive mode: downloading all ${#ARTIFACTS[@]} artifact(s)..."
        print_color $CYAN "TIDL Version: $TIDL_VER | SoC: $SOC"
        echo
        for artifact in "${ARTIFACTS[@]}"; do
            print_color $GREEN "  → $artifact"
        done
        echo
        download_artifacts "${ARTIFACTS[@]}"
        return
    fi
    
    # Interactive mode
    display_artifacts_menu
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
            print_color $BLUE "AudioAI Artifacts Downloader"
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
            print_color $CYAN "  -y, --yes            Non-interactive mode, download all artifacts"
            print_color $CYAN "  -h, --help           Show this help message"
            echo
            print_color $CYAN "Examples:"
            print_color $CYAN "  $0                   # Interactive mode"
            print_color $CYAN "  $0 -y                # Non-interactive, download all artifacts"
            echo
            print_color $CYAN "Available artifacts:"
            for artifact in "${ARTIFACTS[@]}"; do
                print_color $CYAN "  - $artifact"
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
