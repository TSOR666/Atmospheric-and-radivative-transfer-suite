#!/usr/bin/env bash
################################################################################
# Pre-Deployment Validation Script
# Run this before deploying to production to catch common issues
################################################################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

pass() {
    echo -e "${GREEN}[OK]${NC} $1"
}

fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((ERRORS += 1))
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS += 1))
}

info() {
    echo -e "[INFO] $1"
}

case_has_terminator() {
    local label="${1}" file="${2}"
    awk -v label="${label}" '
        BEGIN {found = 0; ok = 0}
        {
            if (!found) {
                if ($0 ~ "^[[:space:]]*" label "\\)") {
                    found = 1
                    next
                }
            } else {
                if ($0 ~ /^[[:space:]]*($|#)/) {
                    next
                }
                if ($0 ~ /^[[:space:]]*;;[[:space:]]*$/) {
                    ok = 1
                    exit
                }
                if ($0 ~ /^[[:space:]]*[^[:space:]]+\\)/) {
                    exit
                }
            }
        }
        END {exit ok ? 0 : 1}
    ' "${file}"
}

echo "=================================="
echo "Pre-Deployment Validation"
echo "=================================="
echo ""

# ============================================================================
# 1. Check deployment script
# ============================================================================

echo "1. Validating deployment script..."

DEPLOY_SCRIPT="deployment_scripts.sh"
if [[ ! -f "${DEPLOY_SCRIPT}" ]]; then
    DEPLOY_SCRIPT="deploy.sh"
fi

if [[ -f "${DEPLOY_SCRIPT}" ]]; then
    pass "${DEPLOY_SCRIPT} exists"
    
    # Check for duplicate functions
    duplicate_count=$(grep -c "^deploy_kubernetes()" "${DEPLOY_SCRIPT}" || echo 0)
    if [[ ${duplicate_count} -eq 1 ]]; then
        pass "No duplicate deploy_kubernetes() definitions"
    else
        fail "Found ${duplicate_count} deploy_kubernetes() definitions (should be 1)"
    fi
    
    # Check for proper case terminators
    if case_has_terminator "kustomize" "${DEPLOY_SCRIPT}" && \
       case_has_terminator "helm" "${DEPLOY_SCRIPT}"; then
        pass "Case statement terminators correct"
    else
        fail "Case statements missing ;; terminators"
    fi
    
    # Check for prune safety
    if grep -q "prune.*--selector" "${DEPLOY_SCRIPT}"; then
        pass "kubectl apply uses label-scoped pruning"
    elif grep -q "prune.*--all" "${DEPLOY_SCRIPT}"; then
        fail "kubectl apply uses dangerous --prune --all"
    else
        warn "Could not verify prune configuration"
    fi
    
    # Check for service URL hostname fallback
    if grep -q "loadBalancer.ingress\[0\].hostname" "${DEPLOY_SCRIPT}"; then
        pass "Service URL checks both IP and hostname"
    else
        warn "Service URL only checks IP (may not work on all clouds)"
    fi
    
    # Check for digest injection
    if grep -q "image-digest-inference.txt" "${DEPLOY_SCRIPT}" && \
       grep -q "sed.*REPLACE_WITH_REAL_DIGEST" "${DEPLOY_SCRIPT}"; then
        pass "Image digest injection implemented"
    else
        fail "Image digest injection not found in render_manifests()"
    fi
    
    # Run shellcheck if available
    if command -v shellcheck &> /dev/null; then
        if shellcheck "${DEPLOY_SCRIPT}"; then
            pass "shellcheck passed"
        else
            fail "shellcheck found issues"
        fi
    else
        warn "shellcheck not installed, skipping syntax check"
    fi
else
    fail "Deployment script not found (checked: deployment_scripts.sh, deploy.sh)"
fi

echo ""

# ============================================================================
# 2. Check Kubernetes manifests
# ============================================================================

echo "2. Validating Kubernetes manifests..."

K8S_FILE=""
for candidate in "kubernetes-production.yaml" "k8s.yaml" "k8s_hardened.txt" "k8s/kubernetes-complete.yaml"; do
    if [[ -f "${candidate}" ]]; then
        K8S_FILE="${candidate}"
        break
    fi
done

if [[ -n "${K8S_FILE}" ]]; then
    pass "Kubernetes manifest found: ${K8S_FILE}"
    
    # Check for placeholder digest
    if grep -q "sha256:REPLACE_WITH_REAL_DIGEST" "${K8S_FILE}"; then
        fail "Image still has placeholder digest (update before deploy)"
        info "  Run: docker inspect IMAGE | jq -r '.[0].RepoDigests[0]'"
    elif grep -q "sha256:abc123" "${K8S_FILE}"; then
        fail "Image has fake digest 'abc123'"
    elif grep -q "@sha256:[a-f0-9]\{64\}" "${K8S_FILE}"; then
        pass "Image uses digest pinning"
    else
        warn "Could not verify image digest format"
    fi
    
    # Check ServiceAccount token policy
    sa_token_count=$(grep -c "automountServiceAccountToken: false" "${K8S_FILE}" || echo 0)
    if [[ ${sa_token_count} -ge 2 ]]; then
        pass "ServiceAccount token mounting disabled (consistent)"
    else
        fail "ServiceAccount token policy inconsistent or not set to false"
    fi
    
    # Check for Helm template artifacts
    if grep -q "{{.*}}" "${K8S_FILE}"; then
        fail "Helm template syntax found in raw manifest"
        info "  Remove: checksum/config or similar annotations"
    else
        pass "No Helm template artifacts"
    fi
    
    # Check for Pod Security Standards
    if grep -q "pod-security.kubernetes.io/enforce" "${K8S_FILE}"; then
        pass "Pod Security Standards labels present"
    else
        warn "Consider adding PSS labels to Namespace"
    fi
    
    # Check for restrictive NetworkPolicy
    if grep -q "ipBlock:" "${K8S_FILE}" && \
       grep -q "except:" "${K8S_FILE}"; then
        pass "NetworkPolicy uses ipBlock with exceptions"
    elif grep -q "namespaceSelector: {}" "${K8S_FILE}"; then
        fail "NetworkPolicy uses overly broad namespaceSelector: {}"
    else
        warn "Could not verify NetworkPolicy configuration"
    fi
    
    # Check for securityContext settings
    if grep -q "runAsNonRoot: true" "${K8S_FILE}" && \
       grep -q "readOnlyRootFilesystem: true" "${K8S_FILE}" && \
       grep -q 'drop: \["ALL"\]' "${K8S_FILE}"; then
        pass "Strong security context configured"
    else
        warn "Security context may not be fully hardened"
    fi
    
    # Check for probes
    if grep -q "startupProbe:" "${K8S_FILE}" && \
       grep -q "livenessProbe:" "${K8S_FILE}" && \
       grep -q "readinessProbe:" "${K8S_FILE}"; then
        pass "All probe types configured (startup, liveness, readiness)"
    else
        warn "Not all probe types found"
    fi
    
    # Check for resource limits
    if grep -q "resources:" "${K8S_FILE}" && \
       grep -q "limits:" "${K8S_FILE}" && \
       grep -q "requests:" "${K8S_FILE}"; then
        pass "Resource requests and limits configured"
    else
        fail "Resource limits not properly configured"
    fi
    
    # Check for duplicate resource definitions
    deployment_count=$(grep -c "^kind: Deployment$" "${K8S_FILE}" || echo 0)
    namespace_count=$(grep -c "^kind: Namespace$" "${K8S_FILE}" || echo 0)
    
    if [[ ${deployment_count} -gt 2 ]]; then
        fail "Found ${deployment_count} Deployment resources (possible duplicates)"
    else
        pass "No duplicate Deployment resources"
    fi
    
    if [[ ${namespace_count} -gt 1 ]]; then
        warn "Found ${namespace_count} Namespace resources (possible duplicates)"
    else
        pass "Single Namespace definition"
    fi
    
    # Run yamllint if available
    if command -v yamllint &> /dev/null; then
        if yamllint "${K8S_FILE}" 2>/dev/null; then
            pass "yamllint passed"
        else
            warn "yamllint found style issues"
        fi
    else
        warn "yamllint not installed, skipping YAML validation"
    fi
    
    # Run kubeconform if available
    if command -v kubeconform &> /dev/null; then
        if kubeconform -strict "${K8S_FILE}" 2>/dev/null; then
            pass "kubeconform validation passed"
        else
            fail "kubeconform found invalid resources"
        fi
    else
        warn "kubeconform not installed, skipping schema validation"
    fi
else
    fail "Kubernetes manifest not found"
fi

echo ""

# ============================================================================
# 3. Check Python API
# ============================================================================

echo "3. Validating Python API..."

API_FILE=""
for candidate in "production_api.py" "api.py" "production_prometheus.py"; do
    if [[ -f "${candidate}" ]]; then
        API_FILE="${candidate}"
        break
    fi
done

if [[ -n "${API_FILE}" ]]; then
    pass "${API_FILE} exists"
    
    # Check multiprocess setup order
    if head -n 30 "${API_FILE}" | grep -q "PROMETHEUS_MULTIPROC_DIR" && \
       head -n 30 "${API_FILE}" | grep -q "os.environ" | head -1; then
        pass "PROMETHEUS_MULTIPROC_DIR set before imports"
    else
        fail "PROMETHEUS_MULTIPROC_DIR not set early enough (must be before prometheus_client import)"
    fi
    
    # Check for private member access
    if grep -q "\._value\.get()" "${API_FILE}" || \
       grep -q "\._metrics\." "${API_FILE}"; then
        fail "Code accesses private Prometheus members"
        info "  Use RequestTracker or REGISTRY.get_sample_value() instead"
    else
        pass "No private Prometheus member access"
    fi
    
    # Check RequestTracker.decrement implementation
    if grep -A5 "async def decrement" "${API_FILE}" | grep -q "self._count"; then
        pass "RequestTracker.decrement properly implemented"
    else
        fail "RequestTracker.decrement appears truncated or missing"
    fi
    
    # Check CORS configuration
    if grep -q 'allow_origins=\["\\*"\]' "${API_FILE}"; then
        fail "CORS allows wildcard origins"
    elif grep -q "allowed_origins" "${API_FILE}" && \
         grep -q "CORS_ORIGINS" "${API_FILE}"; then
        pass "CORS uses environment-driven origins"
    else
        warn "Could not verify CORS configuration"
    fi
    
    # Check for mark_process_dead
    if grep -q "mark_process_dead" "${API_FILE}"; then
        pass "Multiprocess cleanup (mark_process_dead) present"
    else
        warn "mark_process_dead not found (needed for multiprocess mode)"
    fi
    
    # Run ruff/flake8 if available
    if command -v ruff &> /dev/null; then
        if ruff check "${API_FILE}" 2>/dev/null; then
            pass "ruff check passed"
        else
            warn "ruff found linting issues"
        fi
    elif command -v flake8 &> /dev/null; then
        if flake8 "${API_FILE}" --max-line-length=120 2>/dev/null; then
            pass "flake8 passed"
        else
            warn "flake8 found linting issues"
        fi
    else
        warn "No Python linter installed, skipping lint check"
    fi
    
    # Check for type hints with mypy if available
    if command -v mypy &> /dev/null; then
        if mypy "${API_FILE}" --ignore-missing-imports 2>/dev/null; then
            pass "mypy type check passed"
        else
            warn "mypy found type issues"
        fi
    else
        warn "mypy not installed, skipping type check"
    fi
else
    fail "Python API file not found"
fi

echo ""

# ============================================================================
# 4. Environment checks
# ============================================================================

echo "4. Checking environment..."

# Check for secrets file
if [[ -f ".env.production" ]] || [[ -f "secrets.yaml" ]]; then
    pass "Secrets file found"
    
    # Make sure secrets aren't committed
    if git check-ignore .env.production &> /dev/null 2>&1 || \
       git check-ignore secrets.yaml &> /dev/null 2>&1; then
        pass "Secrets file in .gitignore"
    else
        fail "Secrets file not in .gitignore"
    fi
else
    warn "No secrets file found (.env.production or secrets.yaml)"
fi

# Check required tools
for tool in docker kubectl python3; do
    if command -v ${tool} &> /dev/null; then
        pass "${tool} installed"
    else
        fail "${tool} not installed (required)"
    fi
done

for tool in helm kustomize jq; do
    if command -v ${tool} &> /dev/null; then
        pass "${tool} installed"
    else
        warn "${tool} not installed (optional)"
    fi
done

# Check Docker daemon
if docker info &> /dev/null 2>&1; then
    pass "Docker daemon running"
else
    fail "Docker daemon not running"
fi

# Check kubectl access
if kubectl cluster-info &> /dev/null 2>&1; then
    current_context=$(kubectl config current-context)
    pass "kubectl can access cluster: ${current_context}"
    
    # Warn if context looks like production
    if [[ "${current_context}" =~ prod|production ]]; then
        warn "Current context appears to be PRODUCTION"
        info "  Double-check you intend to deploy to: ${current_context}"
    fi
else
    warn "kubectl cannot access cluster"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================

echo "=================================="
echo "Validation Summary"
echo "=================================="
echo ""

if [[ ${ERRORS} -eq 0 ]] && [[ ${WARNINGS} -eq 0 ]]; then
    echo -e "${GREEN}[OK] All checks passed!${NC}"
    echo ""
    echo "Ready to deploy. Recommended next steps:"
    echo "  1. Review changes: git diff"
    echo "  2. Deploy to staging first: ./deployment_scripts.sh deploy staging"
    echo "  3. Run smoke tests: ./deployment_scripts.sh smoke https://staging.yourdomain.com"
    echo "  4. Deploy to production: ./deployment_scripts.sh deploy production"
    exit 0
elif [[ ${ERRORS} -eq 0 ]]; then
    echo -e "${YELLOW}[WARN] ${WARNINGS} warning(s) found${NC}"
    echo ""
    echo "You can proceed, but review the warnings above."
    exit 0
else
    echo -e "${RED}[FAIL] ${ERRORS} error(s) found${NC}"
    if [[ ${WARNINGS} -gt 0 ]]; then
        echo -e "${YELLOW}[WARN] ${WARNINGS} warning(s) found${NC}"
    fi
    echo ""
    echo "Fix the errors above before deploying."
    exit 1
fi

