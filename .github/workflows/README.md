# GitHub Actions Workflows

This directory contains CI/CD workflows for the AudioAI-ModelZoo project.

## build-base-image.yml

Builds and publishes the base Docker image (`audioai-base`) to GitHub Container Registry.

### Overview

- **Purpose**: Automate building the framework-independent base Docker image
- **Trigger**: Manual dispatch only (`workflow_dispatch`)
- **Registry**: GitHub Container Registry (ghcr.io)
- **Target Architecture**: ARM64 (for TI EdgeAI processors)

### How to Use

#### Option 1: GitHub Actions UI (Recommended)

1. Navigate to the **Actions** tab in GitHub
2. Select **Build Base Docker Image** workflow
3. Click **Run workflow** button
4. Configure parameters:
   - **sdk_version**: SDK version tag (e.g., `11.1.0`)
   - **use_native_arm64**: Use native ARM64 runner (requires Team/Enterprise plan)
   - **base_image**: Base Docker image (default: `arm64v8/ubuntu:24.04`)
   - **push_to_registry**: Push to GitHub Container Registry (default: `true`)
5. Click **Run workflow**

#### Option 2: GitHub CLI

```bash
gh workflow run build-base-image.yml \
  --ref master \
  -f sdk_version=11.1.0 \
  -f use_native_arm64=true \
  -f push_to_registry=true
```

#### Option 3: GitHub API

```bash
curl -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  https://api.github.com/repos/$OWNER/$REPO/actions/workflows/build-base-image.yml/dispatches \
  -d '{"ref":"master","inputs":{"sdk_version":"11.1.0","use_native_arm64":"true","push_to_registry":"true"}}'
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `sdk_version` | string | Yes | `11.1.0` | TIDL SDK version for image tagging |
| `use_native_arm64` | boolean | Yes | `true` | Use GitHub-hosted ARM64 runner (faster, requires Team/Enterprise plan) |
| `base_image` | string | No | `arm64v8/ubuntu:24.04` | Base Docker image to build from |
| `push_to_registry` | boolean | Yes | `true` | Push built image to GitHub Container Registry |

### Runner Selection

**Native ARM64 Runners** (`use_native_arm64: true`):
- **Availability**: GitHub Team or Enterprise Cloud plans only
- **Performance**: 3-5x faster than QEMU emulation
- **Build Time**: 10-15 minutes (first build), 3-5 minutes (cached)
- **Cost**: 10x billing multiplier (10 build minutes = 100 billed minutes)
- **Use when**: Frequent builds, fast iteration needed

**QEMU Emulation** (`use_native_arm64: false`):
- **Availability**: All GitHub plans (including Free tier)
- **Performance**: Slower due to emulation overhead
- **Build Time**: 45-90 minutes (first build), 15-30 minutes (cached)
- **Cost**: 1x billing multiplier (standard Linux runner pricing)
- **Use when**: Infrequent builds, cost-conscious, Free tier

### Build Outputs

**Image Tags**:
```
ghcr.io/<username>/audioai-base:11.1.0           # SDK version
ghcr.io/<username>/audioai-base:11.1.0-abc1234   # SDK version + commit SHA
ghcr.io/<username>/audioai-base:latest           # Latest (on default branch)
ghcr.io/<username>/audioai-base:master           # Branch name
```

**Artifacts**:
- Build metadata (SDK version, image digest, runner info)
- Available for 30 days after build

**Summary**:
- Build performance metrics
- Pull command for built image
- Test results (NumPy version, package verification)

### Using Built Images

After the workflow completes, pull and use the image:

```bash
# Pull from registry
docker pull ghcr.io/<username>/audioai-base:11.1.0

# Tag for local use with existing scripts
docker tag ghcr.io/<username>/audioai-base:11.1.0 audioai-base:11.1.0

# Build TI-specific image on top
cd docker
./docker_build_ti.sh
```

### Build Features

**Layer Caching**:
- Uses GitHub Actions cache to speed up rebuilds
- Caches Docker layers between builds
- Significantly reduces build time for incremental changes

**Automatic Testing**:
- Verifies NumPy version (must be 1.26.4 for TIDL compatibility)
- Checks Numba cache presence
- Validates package imports (torch, torchaudio, librosa, soundfile)

**Build Metadata**:
- OCI-compliant labels
- SDK version tracking
- Runner architecture information
- Git commit SHA and build timestamp

### Troubleshooting

**Build fails with "runner not found"**:
- Native ARM64 runners require Team/Enterprise plan
- Solution: Set `use_native_arm64: false` to use QEMU emulation

**Build timeout**:
- QEMU builds can take 45-90 minutes on first run
- Solution: Wait for completion or use native ARM64 runner

**NumPy version mismatch**:
- Test fails if NumPy is not 1.26.4
- Solution: Check `docker/requirements.txt` has `numpy==1.26.4`

**Image not found after build**:
- Check if `push_to_registry: true` was set
- Verify GitHub Container Registry permissions
- Ensure workflow completed successfully

**Cache not working**:
- GitHub Actions cache has size limits (10 GB per repository)
- Old cache entries are automatically evicted
- Check Actions cache settings in repository

### Cost Estimation

**Free Tier (amd64 + QEMU)**:
- 2,000 minutes/month included
- ~5 builds/month: 150-250 minutes
- ✓ Within free tier limits

**Team/Enterprise (ARM64)**:
- Variable minutes depending on plan
- ~5 builds/month: 250-400 billed minutes (25-40 actual minutes)
- 10x multiplier applies

**Optimization Tips**:
- Use path-based triggers sparingly
- Leverage layer caching
- Build only when dependencies change
- Consider self-hosted runners for very frequent builds (>20/month)

### Related Documentation

- [Implementation Plan](../../docs_progress/github-workflow-base-image-plan.md)
- [Plan Updates](../../docs_progress/UPDATES.md)
- [CLAUDE.md](../../CLAUDE.md) - CI/CD section

### Workflow Architecture

```
Trigger (manual dispatch)
  ↓
Detect runner architecture (arm64 vs x86_64)
  ↓
[If x86_64] Set up QEMU for ARM64 emulation
  ↓
Set up Docker Buildx
  ↓
Authenticate to ghcr.io
  ↓
Prepare build context (create proxy/ directory)
  ↓
Build ARM64 image with layer caching
  ↓
[If push enabled] Push to GitHub Container Registry
  ↓
Test image (NumPy version, packages, Numba cache)
  ↓
Generate build summary and artifacts
```

### Security

**Permissions**:
- `contents: read` - Checkout repository
- `packages: write` - Push to GitHub Container Registry
- `id-token: write` - Generate provenance/SBOM (future)

**Authentication**:
- Uses built-in `GITHUB_TOKEN` (no secrets needed)
- Automatically scoped to repository
- Expires after workflow completes

**Image Scanning**:
- Future: Trivy vulnerability scanning
- Future: SBOM generation with Syft
- Future: Cosign image signing

### Contributing

When modifying the workflow:

1. Test changes on a feature branch first
2. Verify YAML syntax: `yamllint build-base-image.yml`
3. Test both runner modes (ARM64 and QEMU)
4. Update this README if parameters or behavior changes
5. Document build time observations
