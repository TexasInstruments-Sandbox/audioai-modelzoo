#!/bin/bash

# Parse command line arguments
NON_INTERACTIVE=false
TIDL_VERSION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes)
            NON_INTERACTIVE=true
            shift
            ;;
        -h|--help)
            # Will be handled in main()
            break
            ;;
        -l|--list)
            # Will be handled in main()
            break
            ;;
        *)
            if [[ -z "$TIDL_VERSION" ]]; then
                TIDL_VERSION="$1"
            fi
            shift
            ;;
    esac
done

# Set default version if not specified
TIDL_VERSION="${TIDL_VERSION:-10_01_00_02}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_URL="http://palserver.dhcp.ti.com/audioai_modelzoo/model_artifacts"
LOCAL_ARTIFACTS_DIR="$SCRIPT_DIR"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

print_color() { echo -e "${1}${2}${NC}"; }
ensure_dir() { [ ! -d "$1" ] && mkdir -p "$1" && print_color $GREEN "Created directory: $1"; }

# Function to get available TIDL versions
get_available_versions() {
    local html=$(wget -qO- "$SERVER_URL/" 2>/dev/null || curl -s "$SERVER_URL/" 2>/dev/null || echo "")
    if [ -n "$html" ]; then
        echo "$html" | grep -oE 'href="[^"]*/"' | sed 's/href="//;s/"//' | grep -E '^[0-9]+_[0-9]+_[0-9]+_[0-9]+/$' | sed 's|/$||' | sort
    fi
}

# Function to show available versions
show_available_versions() {
    print_color $CYAN "Available TIDL versions:"
    local versions=$(get_available_versions)
    if [ -n "$versions" ]; then
        echo "$versions" | while read -r version; do
            print_color $GREEN "  - $version"
        done
    else
        print_color $RED "Could not fetch available versions from server."
    fi
}

# Fetch artifacts recursively
fetch_artifacts() {
    print_color $CYAN "Fetching available artifacts for TIDL version $TIDL_VERSION..." >&2
    local artifacts=()
    
    explore() {
        local url="$1" path="$2"
        local html=$(wget -qO- "$url" 2>/dev/null || curl -s "$url" 2>/dev/null || echo "")
        [ -z "$html" ] && return
        
        local links=$(echo "$html" | grep -oE 'href="[^"]*"' | sed 's/href="//;s/"//' | grep -v '^[/.?]' | grep -v '^\.\.' | grep -v '^$')
        
        for link in $links; do
            local current_path="${path:+$path/}${link%/}"
            
            if [[ "$link" == */ ]]; then
                # Directory
                local dir_name="${link%/}"
                
                # If we're at the root level, only explore the specified TIDL version directory
                if [[ -z "$path" ]]; then
                    if [[ "$dir_name" == "$TIDL_VERSION" ]]; then
                        explore "$url$link" "$current_path"
                    fi
                else
                    # If we're already inside the TIDL version directory, check for artifacts
                    if [[ "$dir_name" =~ ^(h-|encoder_|vggish|gtcrn|whisper|espcn) ]] || [[ "$current_path" =~ (int8|int16|fp32)$ ]]; then
                        local subhtml=$(wget -qO- "$url$link" 2>/dev/null || curl -s "$url$link" 2>/dev/null || echo "")
                        if echo "$subhtml" | grep -qE 'href="[^"]*\.(bin|txt|csv|svg|config)"'; then
                            artifacts+=("$current_path/")
                        fi
                    fi
                    # Continue recursive exploration
                    explore "$url$link" "$current_path"
                fi
            elif [[ "$link" =~ \.(tar\.gz|tgz|zip|tar)$ ]]; then
                # Archive file - only add if we're inside the TIDL version directory
                if [[ -n "$path" ]] && [[ "$path" =~ ^$TIDL_VERSION ]]; then
                    artifacts+=("$current_path")
                fi
            fi
        done
    }
    
    explore "$SERVER_URL/" ""
    printf '%s\n' "${artifacts[@]}"
}

# Download directory recursively
download_dir() {
    local remote_path="$1" local_base="$2"
    local total_files=0 downloaded_files=0
    
    # Function to count total files first
    count_files() {
        local url="$1"
        local html=$(wget -qO- "$url" 2>/dev/null || curl -s "$url" 2>/dev/null || echo "")
        [ -z "$html" ] && return
        
        local files=$(echo "$html" | grep -oE 'href="[^"]*"' | sed 's/href="//;s/"//' | grep -v '^[/.?]' | grep -v '^\.\.' | grep -v '^$' | grep -v '/$')
        total_files=$((total_files + $(echo "$files" | wc -w)))
        
        local subdirs=$(echo "$html" | grep -oE 'href="[^"]*/"' | sed 's/href="//;s/"//' | grep -v '^[/.?]' | grep -v '^\.\.' | grep -v '^$')
        for subdir in $subdirs; do
            count_files "$url$subdir"
        done
    }
    
    # Function to download files recursively
    download_files() {
        local url="$1" path="$2"
        local html=$(wget -qO- "$url" 2>/dev/null || curl -s "$url" 2>/dev/null || echo "")
        [ -z "$html" ] && return
        
        local files=$(echo "$html" | grep -oE 'href="[^"]*"' | sed 's/href="//;s/"//' | grep -v '^[/.?]' | grep -v '^\.\.' | grep -v '^$' | grep -v '/$')
        for file in $files; do
            local local_file="$local_base/${path%/}/$file"
            ensure_dir "$(dirname "$local_file")" >/dev/null 2>&1
            
            if wget -q -O "$local_file" "$url$file" 2>/dev/null || curl -s -o "$local_file" "$url$file" 2>/dev/null; then
                downloaded_files=$((downloaded_files + 1))
            fi
            printf "\r  Progress: %d/%d files downloaded" "$downloaded_files" "$total_files"
        done
        
        local subdirs=$(echo "$html" | grep -oE 'href="[^"]*/"' | sed 's/href="//;s/"//' | grep -v '^[/.?]' | grep -v '^\.\.' | grep -v '^$')
        for subdir in $subdirs; do
            download_files "$url$subdir" "$path$subdir"
        done
    }
    
    local url="$SERVER_URL/$remote_path"
    if [ -z "$(wget -qO- "$url" 2>/dev/null || curl -s "$url" 2>/dev/null)" ]; then
        print_color $RED "Failed to access: $remote_path"
        return 0
    fi
    
    # Count total files first
    count_files "$url"
    
    # Download all files with progress
    download_files "$url" "$remote_path"
    echo  # New line after progress
    
    if [ $downloaded_files -eq $total_files ] && [ $total_files -gt 0 ]; then
        print_color $GREEN "✓ Successfully downloaded all $total_files files"
    elif [ $downloaded_files -gt 0 ]; then
        print_color $YELLOW "⚠ Downloaded $downloaded_files/$total_files files"
    else
        print_color $RED "✗ Failed to download any files"
    fi
}

# Extract archive
extract() {
    local file="$1" dir="$2"
    print_color $CYAN "Extracting: $(basename "$file")"
    
    case "$file" in
        *.tar.gz|*.tgz) 
            if tar -xzf "$file" -C "$dir" 2>/dev/null; then
                print_color $GREEN "✓ Extracted: $(basename "$file")"
                rm "$file" 2>/dev/null
                print_color $YELLOW "Removed archive: $(basename "$file")"
            else
                print_color $RED "✗ Failed to extract: $(basename "$file")"
            fi ;;
        *.zip) 
            if unzip -q "$file" -d "$dir" 2>/dev/null; then
                print_color $GREEN "✓ Extracted: $(basename "$file")"
                rm "$file" 2>/dev/null
                print_color $YELLOW "Removed archive: $(basename "$file")"
            else
                print_color $RED "✗ Failed to extract: $(basename "$file")"
            fi ;;
        *.tar) 
            if tar -xf "$file" -C "$dir" 2>/dev/null; then
                print_color $GREEN "✓ Extracted: $(basename "$file")"
                rm "$file" 2>/dev/null
                print_color $YELLOW "Removed archive: $(basename "$file")"
            else
                print_color $RED "✗ Failed to extract: $(basename "$file")"
            fi ;;
        *) print_color $YELLOW "No extraction needed"; return ;;
    esac
}

# Download artifacts
download() {
    local artifacts=("$@")
    echo; print_color $CYAN "Starting download of ${#artifacts[@]} artifact(s)..."
    
    for ((i=0; i<${#artifacts[@]}; i++)); do
        local artifact="${artifacts[i]}"
        local artifact_num=$((i+1))
        
        echo; print_color $BLUE "[$artifact_num/${#artifacts[@]}] Processing: $artifact"
        
        if [[ "$artifact" == */ ]]; then
            download_dir "$artifact" "$LOCAL_ARTIFACTS_DIR"
        else
            local local_path="$LOCAL_ARTIFACTS_DIR/$artifact"
            local remote_url="$SERVER_URL/$artifact"
            
            ensure_dir "$(dirname "$local_path")" >/dev/null 2>&1
            
            local download_success=0
            if wget -q -O "$local_path" "$remote_url" 2>/dev/null; then
                download_success=1
            elif curl -s -o "$local_path" "$remote_url" 2>/dev/null; then
                download_success=1
            fi
            
            if [ $download_success -eq 1 ]; then
                print_color $GREEN "✓ Downloaded: $artifact"
                extract "$local_path" "$(dirname "$local_path")"
            else
                print_color $RED "✗ Failed to download: $artifact"
            fi
        fi
    done
    
    echo; print_color $GREEN "Download completed!"
    print_color $CYAN "Artifacts saved to: $LOCAL_ARTIFACTS_DIR"
}

# Main menu
main() {
    # Check for help flag
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        print_color $BLUE "AudioAI Artifacts Downloader"
        print_color $BLUE "============================"
        echo
        print_color $YELLOW "Usage: $0 [OPTIONS] [TIDL_VERSION]"
        print_color $YELLOW "       $0 -h|--help"
        print_color $YELLOW "       $0 -l|--list"
        echo
        print_color $CYAN "Options:"
        print_color $CYAN "  TIDL_VERSION         Specify TIDL version (default: 10_01_00_02)"
        print_color $CYAN "  -y, --yes            Non-interactive mode, download all artifacts"
        print_color $CYAN "  -h, --help           Show this help message"
        print_color $CYAN "  -l, --list           List available TIDL versions"
        echo
        print_color $CYAN "Examples:"
        print_color $CYAN "  $0                        # Interactive mode with default version"
        print_color $CYAN "  $0 11_00_06_00            # Interactive mode with specific version"
        print_color $CYAN "  $0 -y                     # Non-interactive, download all (default version)"
        print_color $CYAN "  $0 -y 11_00_06_00         # Non-interactive, download all (specific version)"
        print_color $CYAN "  $0 11_00_06_00 -y         # Same as above (order doesn't matter)"
        print_color $CYAN "  $0 -l                     # List available versions"
        exit 0
    fi
    
    # Check for list flag
    if [[ "$1" == "-l" || "$1" == "--list" ]]; then
        show_available_versions
        exit 0
    fi
    
    print_color $BLUE "AudioAI Artifacts Downloader (TIDL Version: $TIDL_VERSION)"
    print_color $BLUE "============================================================"
    echo
    
    # Check tools
    if ! command -v wget &>/dev/null && ! command -v curl &>/dev/null; then
        print_color $RED "Error: Neither wget nor curl is available."
        exit 1
    fi
    
    # Fetch artifacts
    print_color $CYAN "Fetching available artifacts for TIDL version $TIDL_VERSION..."
    local artifacts_list=$(fetch_artifacts 2>/dev/null)
    
    if [ -z "$artifacts_list" ]; then
        print_color $RED "No artifacts found for TIDL version $TIDL_VERSION."
        echo
        print_color $YELLOW "Usage: $0 [OPTIONS] [TIDL_VERSION]"
        print_color $YELLOW "Use '$0 -l' to list available TIDL versions"
        echo
        show_available_versions
        exit 1
    fi
    
    readarray -t artifacts <<< "$artifacts_list"
    local filtered=()
    for artifact in "${artifacts[@]}"; do
        [[ -n "$artifact" && ! "$artifact" =~ ^[[:space:]]*$ && ! "$artifact" =~ \\033 ]] && filtered+=("$artifact")
    done
    
    [ ${#filtered[@]} -eq 0 ] && { 
        print_color $RED "No artifacts found for TIDL version $TIDL_VERSION."
        echo
        print_color $YELLOW "Usage: $0 [OPTIONS] [TIDL_VERSION]"
        print_color $YELLOW "Use '$0 -l' to list available TIDL versions"
        echo
        show_available_versions
        exit 1
    }
    
    # Non-interactive mode: download all artifacts
    if [ "$NON_INTERACTIVE" = true ]; then
        print_color $CYAN "Non-interactive mode: downloading all ${#filtered[@]} artifact(s)..."
        echo
        for ((i=0; i<${#filtered[@]}; i++)); do
            local artifact="${filtered[i]}" type=""
            [[ "$artifact" == */ ]] && type=" [DIR]"
            [[ "$artifact" =~ \.(tar\.gz|tgz|zip|tar)$ ]] && type=" [ARCHIVE]"
            print_color $GREEN "  ✓ $artifact$type"
        done
        echo
        download "${filtered[@]}"
        return
    fi
    
    # Interactive mode
    # Initialize selection (all selected)
    local selected=()
    for ((i=0; i<${#filtered[@]}; i++)); do selected[i]=1; done
    
    # Interactive menu
    while true; do
        clear
        print_color $BLUE "════════════════════════════════════════════════════════════════════"
        print_color $BLUE "           AudioAI Artifacts Downloader (v$TIDL_VERSION)            "
        print_color $BLUE "════════════════════════════════════════════════════════════════════"
        echo; print_color $CYAN "Available Artifacts (✓ = selected, ✗ = deselected):"; echo
        
        for ((i=0; i<${#filtered[@]}; i++)); do
            local artifact="${filtered[i]}" status="✗" color=$RED type=""
            [ "${selected[i]}" -eq 1 ] && { status="✓"; color=$GREEN; }
            [[ "$artifact" == */ ]] && type=" [DIR]"
            [[ "$artifact" =~ \.(tar\.gz|tgz|zip|tar)$ ]] && type=" [ARCHIVE]"
            printf "${color}[%2d] %s %s%s${NC}\n" $((i+1)) "$status" "$artifact" "$type"
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
                if [ $idx -ge 0 ] && [ $idx -lt ${#filtered[@]} ]; then
                    selected[idx]=$((1 - ${selected[idx]}))
                fi ;;
            a|A) for ((i=0; i<${#filtered[@]}; i++)); do selected[i]=1; done ;;
            n|N) for ((i=0; i<${#filtered[@]}; i++)); do selected[i]=0; done ;;
            d|D)
                local sel=()
                for ((i=0; i<${#filtered[@]}; i++)); do
                    [ "${selected[i]}" -eq 1 ] && sel+=("${filtered[i]}")
                done
                if [ ${#sel[@]} -eq 0 ]; then
                    print_color $RED "No artifacts selected!"
                    read -p "Press Enter to continue..."
                else
                    download "${sel[@]}"; return
                fi ;;
            q|Q) print_color $YELLOW "Goodbye!"; exit 0 ;;
            *) print_color $RED "Invalid choice."; sleep 1 ;;
        esac
    done
}

main "$@"