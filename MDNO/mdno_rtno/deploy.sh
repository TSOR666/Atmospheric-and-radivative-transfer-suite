#!/usr/bin/env bash
################################################################################
# Production Deployment Scripts
# Version: 3.0
# Fully deduplicated, production-hardened deployment automation
################################################################################

set -Eeuo pipefail
IFS=$'\n\t'
umask 027

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
readonly VERSION="3.0.0"

trap 'echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO} failed with exit code $?" >&2; exit 1' ERR

# Configuration
PROJECT_NAME="${PROJECT_NAME:-atmospheric-models}"
REGISTRY="${DOCKER_REGISTRY:-ghcr.io/your-org}"
NAMESPACE="${K8S_NAMESPACE:-atmospheric-models}"
STRATEGY="${DEPLOY_STRATEGY:-kubectl}"

# Colors
if [[ -t 2 ]] && [[ -z "${NO_COLOR:-}" ]]; then
    readonly RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m' BLUE='\033[0;34m' NC='\033[0m'
else
    readonly RED='' GREEN='' YELLOW='' BLUE='' NC=''
fi

# Flags
DRY_RUN=false
VERBOSE=false
SKIP_TESTS=false
SKIP_PUSH=false
SKIP_SCAN=false
FORCE=false

################################################################################
# Logging
################################################################################

log() { printf '%s[%s]%s %s\n' "${GREEN}" "$(date -Is)" "${NC}" "$*" >&2; }
error() { printf '%s[ERROR]%s %s\n' "${RED}" "${NC}" "$*" >&2; }
warning() { printf '%s[WARN]%s %s\n' "${YELLOW}" "${NC}" "$*" >&2; }
info() { printf '%s[INFO]%s %s\n' "${BLUE}" "${NC}" "$*" >&2; }
verbose() { [[ "${VERBOSE}" == "true" ]] && printf '%s[DEBUG]%s %s\n' "${BLUE}" "${NC}" "$*" >&2 || true; }

run() {
    verbose "+ $*"
    [[ "${DRY_RUN}" == "true" ]] && { info "[DRY-RUN] Would execute: $*"; return 0; }
    "$@"
}

################################################################################
# Usage
################################################################################

usage() {
    cat <<EOF
Usage: ${SCRIPT_NAME} [OPTIONS] COMMAND [ARGS...]

Production deployment automation.

COMMANDS:
    deploy <env>          Full deployment (local|staging|production)
    build                 Build Docker images
    test                  Run test suite
    push                  Push images to registry
    k8s <env>             Deploy to Kubernetes
    rollback              Rollback deployment
    health <url>          Health check
    smoke <url>           Smoke tests
    preflight             Preflight checks
    cleanup               Clean old resources

OPTIONS:
    -h, --help            Show help
    -v, --verbose         Verbose logging
    -n, --dry-run         Dry run mode
    -f, --force           Force operations
    --skip-tests          Skip tests
    --skip-push           Skip push
    --skip-scan           Skip security scan
    --env FILE            Load environment
    --strategy TYPE       kubectl|helm|kustomize
    --version             Show version

EXAMPLES:
    ${SCRIPT_NAME} -v deploy production
    ${SCRIPT_NAME} --dry-run deploy staging
    ${SCRIPT_NAME} --skip-tests build push

VERSION: ${VERSION}
EOF
}

################################################################################
# Validation
################################################################################

validate_cluster() {
    case "${1}" in
        local|dev|staging|production) return 0 ;;
        *) error "Invalid cluster: ${1}"; return 1 ;;
    esac
}

validate_namespace() {
    [[ "${1}" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]] || {
        error "Invalid namespace: ${1}"; return 1
    }
}

################################################################################
# Preflight
################################################################################

check_command() {
    command -v "${1}" &> /dev/null && verbose "[OK] ${1} found" || {
        error "[FAIL] ${1} not installed"; return 1
    }
}

preflight_checks() {
    log "Running preflight checks..."
    
    local failed=false
    for cmd in docker git kubectl; do
        check_command "${cmd}" || failed=true
    done
    
    docker info &> /dev/null || { error "Docker daemon not running"; failed=true; }
    
    if [[ "${failed}" == "true" ]]; then
        error "Preflight checks failed"
        return 1
    fi
    
    log "[OK] Preflight checks passed"
}

################################################################################
# Retry Logic
################################################################################

retry() {
    local max="${1}" delay="${2}"; shift 2
    local attempt=1
    
    while [[ ${attempt} -le ${max} ]]; do
        "$@" && return 0
        
        [[ ${attempt} -lt ${max} ]] && {
            warning "Retry ${attempt}/${max} in ${delay}s..."
            sleep "${delay}"
            delay=$((delay * 2))
        }
        ((attempt++))
    done
    
    error "Failed after ${max} attempts"
    return 1
}

################################################################################
# Docker Auth
################################################################################

docker_login_safe() {
    [[ -n "${DOCKER_USERNAME:-}" ]] && [[ -n "${DOCKER_PASSWORD:-}" ]] || {
        warning "Docker credentials not set"
        return 1
    }
    
    info "Logging into ${REGISTRY}..."
    echo "${DOCKER_PASSWORD}" | run docker login "${REGISTRY}" \
        -u "${DOCKER_USERNAME}" --password-stdin
}

################################################################################
# Build
################################################################################

build_image() {
    local type="${1}" tag="${2:-latest}"
    local dockerfile="Dockerfile.${type}"
    local image="${REGISTRY}/${PROJECT_NAME}:${type}-${tag}"
    
    log "Building ${type} image..."
    
    [[ -f "${dockerfile}" ]] || { error "Missing ${dockerfile}"; return 1; }
    
    local git_sha
    git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    
    run docker build \
        -f "${dockerfile}" \
        -t "${image}" \
        -t "${REGISTRY}/${PROJECT_NAME}:${type}-latest" \
        --build-arg VERSION="${tag}" \
        --build-arg GIT_SHA="${git_sha}" \
        --build-arg BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
        --label "org.opencontainers.image.version=${tag}" \
        --label "org.opencontainers.image.revision=${git_sha}" \
        .
    
    # Capture digest
    local digest
    digest=$(docker inspect --format='{{index .Id}}' "${image}")
    echo "${image}@${digest}" > ".image-digest-${type}.txt"
    
    log "[OK] Built ${image}"
}

build_all_images() {
    local tag="${1:-$(git rev-parse --short HEAD)}"
    build_image "training" "${tag}"
    build_image "inference" "${tag}"
}

################################################################################
# Security Scanning
################################################################################

generate_sbom() {
    [[ "${SKIP_SCAN}" == "true" ]] && { warning "Skipping SBOM"; return 0; }
    
    if ! command -v syft &> /dev/null; then
        warning "syft not installed, skipping SBOM"
        return 0
    fi
    
    log "Generating SBOM..."
    local image="${REGISTRY}/${PROJECT_NAME}:inference-latest"
    run syft "${image}" -o spdx-json > sbom.json
    log "[OK] SBOM generated: sbom.json"
}

scan_vulnerabilities() {
    [[ "${SKIP_SCAN}" == "true" ]] && { warning "Skipping scan"; return 0; }
    
    if ! command -v grype &> /dev/null; then
        warning "grype not installed, skipping scan"
        return 0
    fi
    
    log "Scanning for vulnerabilities..."
    local image="${REGISTRY}/${PROJECT_NAME}:inference-latest"
    
    if run grype "${image}" --fail-on high; then
        log "[OK] No high/critical vulnerabilities"
    else
        error "Vulnerabilities found"
        [[ "${FORCE}" == "true" ]] || return 1
    fi
}

sign_image() {
    [[ "${SKIP_SCAN}" == "true" ]] && { warning "Skipping signing"; return 0; }
    
    if ! command -v cosign &> /dev/null; then
        warning "cosign not installed, skipping signing"
        return 0
    fi
    
    log "Signing image..."
    local image="${REGISTRY}/${PROJECT_NAME}:inference-latest"
    
    if [[ -n "${COSIGN_KEY:-}" ]]; then
        run cosign sign --key "${COSIGN_KEY}" "${image}"
        log "[OK] Image signed"
    else
        warning "COSIGN_KEY not set, skipping signing"
    fi
}

################################################################################
# Test
################################################################################

run_tests() {
    [[ "${SKIP_TESTS}" == "true" ]] && { warning "Skipping tests"; return 0; }

    log "Running test suite..."
    local image="${REGISTRY}/${PROJECT_NAME}:training-latest"
    local fallback_cmd='if [ -d tests ] && [ "$(ls -A tests)" ]; then pytest tests/ -v --tb=short --maxfail=3; else python -m mdno.demo && python -m rtno.demo; fi'

    run docker run --rm \
        --gpus all \
        -v "${SCRIPT_DIR}:/workspace" \
        -w /workspace \
        "${image}" \
        bash -lc "${fallback_cmd}"

    log "[OK] Tests passed"
}

################################################################################
# Push
################################################################################

push_image() {
    [[ "${SKIP_PUSH}" == "true" ]] && { warning "Skipping push"; return 0; }
    
    log "Pushing ${1}..."
    retry 3 5 docker push "${1}"
}

push_all_images() {
    local tag="${1:-latest}"
    
    docker_login_safe || return 1
    
    push_image "${REGISTRY}/${PROJECT_NAME}:training-${tag}"
    push_image "${REGISTRY}/${PROJECT_NAME}:training-latest"
    push_image "${REGISTRY}/${PROJECT_NAME}:inference-${tag}"
    push_image "${REGISTRY}/${PROJECT_NAME}:inference-latest"
    
    log "[OK] All images pushed"
}

################################################################################
# Kubernetes Deployment
################################################################################

render_manifests() {
    local env="${1}" output_dir="${2:-.deploy}"
    
    log "Rendering manifests for ${env}..."
    mkdir -p "${output_dir}"
    
    case "${STRATEGY}" in
        kustomize)
            [[ -d "kustomize/overlays/${env}" ]] || {
                error "Overlay not found: kustomize/overlays/${env}"
                return 1
            }
            run kustomize build "kustomize/overlays/${env}" > "${output_dir}/manifests.yaml"
            ;;
        helm)
            local values_file="helm/values-${env}.yaml"
            [[ -f "${values_file}" ]] || warning "No values file: ${values_file}"
            
            run helm template "${PROJECT_NAME}" ./helm \
                --namespace "${NAMESPACE}" \
                ${values_file:+--values "${values_file}"} \
                > "${output_dir}/manifests.yaml"
            ;;
        *)
            local manifest="k8s/kubernetes-${env}.yaml"
            [[ -f "${manifest}" ]] || manifest="k8s/kubernetes-complete.yaml"
            [[ -f "${manifest}" ]] || { error "No manifest found"; return 1; }
            cp "${manifest}" "${output_dir}/manifests.yaml"
            ;;
    esac
    
    # Inject digest
    if [[ -f ".image-digest-inference.txt" ]]; then
        local full_ref
        full_ref=$(cat .image-digest-inference.txt)
        local digest="${full_ref##*@}"
        
        if [[ -n "${digest}" ]] && [[ "${digest}" =~ ^sha256: ]]; then
            info "Injecting digest: ${digest}"
            sed -i.bak "s|sha256:REPLACE_WITH_REAL_DIGEST|${digest}|g" \
                "${output_dir}/manifests.yaml"
            rm -f "${output_dir}/manifests.yaml.bak"
        fi
    fi
    
    verbose "[OK] Manifests rendered"
}

deploy_kubernetes() {
    local env="${1}"
    
    log "Deploying to Kubernetes (${env})..."
    
    validate_cluster "${env}"
    validate_namespace "${NAMESPACE}"
    
    # Setup error trap for auto-rollback
    local previous_trap
    previous_trap=$(trap -p ERR | sed "s/trap -- '\(.*\)' ERR/\1/")
    trap 'error "Deployment failed, initiating rollback..."; rollback_deployment; exit 1' ERR
    
    run kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | \
        run kubectl apply -f -
    
    local manifest_dir=".deploy-${env}"
    render_manifests "${env}" "${manifest_dir}"
    
    # Show diff before applying (dry-run)
    info "Previewing changes..."
    kubectl diff \
        --namespace="${NAMESPACE}" \
        --filename="${manifest_dir}/manifests.yaml" \
        --server-side 2>/dev/null || true
    
    # Apply with server-side apply and label-scoped pruning
    run kubectl apply \
        --namespace="${NAMESPACE}" \
        --filename="${manifest_dir}/manifests.yaml" \
        --server-side \
        --force-conflicts \
        --prune \
        --selector="app=mdno"
    
    log "Waiting for rollout..."
    run kubectl rollout status deployment/mdno-inference \
        --namespace="${NAMESPACE}" --timeout=5m
    
    # Restore previous trap
    if [[ -n "${previous_trap}" ]]; then
        trap "${previous_trap}" ERR
    else
        trap - ERR
    fi
    
    log "[OK] Deployment complete"
    
    # Get service URL
    local url
    url=$(kubectl get svc mdno-inference-service -n "${NAMESPACE}" \
        -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
    [[ -z "${url}" ]] && url=$(kubectl get svc mdno-inference-service -n "${NAMESPACE}" \
        -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
    
    [[ -n "${url}" ]] && info "Service: http://${url}" || info "URL pending"
}

################################################################################
# Health Checks
################################################################################

health_check() {
    local url="${1}"
    log "Health check: ${url}..."
    
    local max=30 attempt=1
    while [[ ${attempt} -le ${max} ]]; do
        curl -sf --max-time 5 "${url}/health" > /dev/null 2>&1 && {
            log "[OK] Healthy"
            curl -s "${url}/health" | python3 -m json.tool || true
            return 0
        }
        verbose "Attempt ${attempt}/${max}..."
        sleep 10
        ((attempt++))
    done
    
    error "Health check failed"
    return 1
}

smoke_tests() {
    local url="${1}"
    log "Smoke tests: ${url}..."
    
    curl -sf "${url}/health" > /dev/null || { error "Health failed"; return 1; }
    curl -sf "${url}/metrics" | grep -q "http_requests_total" || {
        error "Metrics failed"; return 1
    }
    
    log "[OK] Smoke tests passed"
}

################################################################################
# Rollback
################################################################################

rollback_deployment() {
    log "Rolling back..."
    
    if [[ "${STRATEGY}" == "helm" ]]; then
        run helm rollback "${PROJECT_NAME}" --namespace="${NAMESPACE}"
    else
        run kubectl rollout undo deployment/mdno-inference --namespace="${NAMESPACE}"
    fi
    
    log "[OK] Rollback complete"
}

################################################################################
# Cleanup
################################################################################

cleanup_resources() {
    log "Cleaning up..."
    
    docker ps -q --filter "name=${PROJECT_NAME}" | xargs -r docker stop 2>/dev/null || true
    docker images "${REGISTRY}/${PROJECT_NAME}" --format "{{.ID}}" | tail -n +6 | \
        xargs -r docker rmi 2>/dev/null || true
    run docker volume prune -f
    
    log "[OK] Cleanup complete"
}

################################################################################
# Full Deploy
################################################################################

deploy_full() {
    local env="${1}"
    log "Full deployment to ${env}..."
    
    preflight_checks || return 1
    
    local tag="$(date +%Y%m%d)-$(git rev-parse --short HEAD)"
    build_all_images "${tag}"
    
    generate_sbom
    scan_vulnerabilities
    sign_image
    
    run_tests
    push_all_images "${tag}"
    
    deploy_kubernetes "${env}"
    
    sleep 10
    local url
    if [[ "${env}" == "local" ]]; then
        url="http://localhost:8000"
    else
        url=$(kubectl get svc mdno-inference-service -n "${NAMESPACE}" \
            -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
        [[ -z "${url}" ]] && url=$(kubectl get svc mdno-inference-service -n "${NAMESPACE}" \
            -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "")
        [[ -n "${url}" ]] && url="http://${url}" || {
            warning "URL not available"
            return 0
        }
    fi
    
    health_check "${url}"
    smoke_tests "${url}"
    
    log "[OK] Deployment successful!"
}

################################################################################
# Parse Args
################################################################################

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "${1}" in
            -h|--help) usage; exit 0 ;;
            -v|--verbose) VERBOSE=true; shift ;;
            -n|--dry-run) DRY_RUN=true; shift ;;
            -f|--force) FORCE=true; shift ;;
            --skip-tests) SKIP_TESTS=true; shift ;;
            --skip-push) SKIP_PUSH=true; shift ;;
            --skip-scan) SKIP_SCAN=true; shift ;;
            --env) source "${2}"; shift 2 ;;
            --strategy) STRATEGY="${2}"; shift 2 ;;
            --namespace) NAMESPACE="${2}"; shift 2 ;;
            --registry) REGISTRY="${2}"; shift 2 ;;
            --version) echo "${VERSION}"; exit 0 ;;
            -*) error "Unknown option: ${1}"; usage; exit 1 ;;
            *) break ;;
        esac
    done
    echo "$@"
}

################################################################################
# Main
################################################################################

main() {
    local args
    args=$(parse_args "$@")
    eval set -- "${args}"
    
    local command="${1:-help}"
    shift || true
    
    case "${command}" in
        deploy) deploy_full "$@" ;;
        build) preflight_checks; build_all_images "$@" ;;
        test) preflight_checks; run_tests ;;
        push) push_all_images "$@" ;;
        k8s|kubernetes) deploy_kubernetes "$@" ;;
        rollback) rollback_deployment ;;
        health) health_check "$@" ;;
        smoke) smoke_tests "$@" ;;
        preflight) preflight_checks ;;
        cleanup) cleanup_resources ;;
        help|--help|-h) usage ;;
        *) error "Unknown command: ${command}"; usage; exit 1 ;;
    esac
}

[[ "${BASH_SOURCE[0]}" == "${0}" ]] && main "$@"

