#!/bin/zsh
#
# git_update.sh — 一键同步本地改动到 GitHub
# 用法：./git_update.sh "提交说明"
#       不传参数时自动生成时间戳作为提交信息
#

set -o pipefail

REMOTE="origin"
BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || echo 'main')"

# 颜色
RED='\033[31m'; GREEN='\033[32m'; YELLOW='\033[33m'; NC='\033[0m'
info()  { echo "${GREEN}[✓]${NC} $1"; }
warn()  { echo "${YELLOW}[!]${NC} $1"; }
error() { echo "${RED}[✗]${NC} $1"; }

# ===================== 前置检查 =====================

cd "$(dirname "$0")" || exit 1

if [[ ! -d .git ]]; then
    error "当前目录不是 Git 仓库"
    exit 1
fi

REMOTE_URL=$(git remote get-url origin 2>/dev/null) || {
    error "没有配置远程仓库 origin"
    exit 1
}

info "仓库: $REMOTE_URL"
info "分支: $BRANCH"

# ===================== 提交信息 =====================

if [[ $# -ge 1 ]]; then
    MSG="$1"
else
    MSG="Auto update $(date '+%Y-%m-%d %H:%M')"
fi

# ===================== 网络连通性检查 =====================

check_ssh() {
    # ssh -T 返回 0=认证成功, 1=密钥被拒(但连接通了), 255=连接失败
    ssh -T -o ConnectTimeout=5 -o BatchMode=yes git@github.com 2>/dev/null
    local rc=$?
    if [[ $rc -eq 255 ]]; then
        return 1
    fi
    return 0
}

if [[ "$REMOTE_URL" == git@* ]]; then
    info "检测到 SSH 远程，测试连接 …"
    if check_ssh; then
        info "SSH 连接正常"
    else
        error "SSH 连接失败！"
        echo ""
        echo "  可能的原因和解决办法："
        echo ""
        echo "  [1] SSH 密钥未加载到 agent"
        echo "      运行: ssh-add --apple-use-keychain ~/.ssh/id_ed25519"
        echo ""
        echo "  [2] 网络无法访问 ssh.github.com:443"
        echo "      尝试切换为 HTTPS 远程地址:"
        echo "      git remote set-url origin https://github.com/Kevin-Cao96/neural-networks.git"
        echo "      然后在 GitHub 生成 Personal Access Token, 用 token 作为密码"
        echo ""
        echo "  [3] 密钥未添加到 GitHub 账户"
        echo "      检查: ssh -T git@github.com"
        echo "      添加: cat ~/.ssh/id_ed25519.pub → GitHub Settings → SSH keys"
        exit 1
    fi
elif [[ "$REMOTE_URL" == https://* ]]; then
    if ! curl -s -o /dev/null --connect-timeout 5 https://github.com; then
        error "无法访问 GitHub，请检查网络/代理设置"
        exit 1
    fi
    info "HTTPS 连接正常"
fi

# ===================== 暂存本地改动 =====================

HAS_CHANGE=$(git status --porcelain)
STASHED=0

if [[ -n "$HAS_CHANGE" ]]; then
    warn "检测到本地未提交改动，自动暂存 …"
    git stash push -m "git_update auto stash $(date +%s)" || {
        error "暂存失败"
        exit 1
    }
    STASHED=1
fi

# ===================== 拉取远程 (rebase) =====================

info "拉取远程最新代码 …"
if ! git pull "$REMOTE" "$BRANCH" --rebase 2>/dev/null; then
    # 判断是网络错误还是冲突
    if ! git rev-list --count "HEAD..$REMOTE/$BRANCH" 2>/dev/null >/dev/null; then
        error "无法连接远程仓库，请检查网络"
    else
        error "合并冲突，请手动解决后执行: git rebase --continue"
    fi
    [[ $STASHED -eq 1 ]] && git stash pop 2>/dev/null || true
    exit 1
fi

# ===================== 恢复暂存 =====================

if [[ $STASHED -eq 1 ]]; then
    info "恢复暂存的本地改动 …"
    git stash pop 2>/dev/null || warn "暂存恢复失败（可能已被合并）"
fi

# ===================== 提交 =====================

git add -A

if git diff --cached --quiet; then
    info "没有新的文件变更，跳过提交"
else
    info "提交: $MSG"
    git commit -m "$MSG" || {
        error "提交失败"
        exit 1
    }
fi

# ===================== 检查推送状态 =====================

AHEAD=$(git rev-list --count "$REMOTE/$BRANCH..HEAD" 2>/dev/null || echo 0)
BEHIND=$(git rev-list --count "HEAD..$REMOTE/$BRANCH" 2>/dev/null || echo 0)

if [[ $AHEAD -eq 0 ]]; then
    info "本地和远程一致，无需推送"
    exit 0
fi

if [[ $BEHIND -gt 0 ]]; then
    error "本地落后远程 ${BEHIND} 个提交，请先手动 git pull"
    exit 1
fi

# ===================== 推送 =====================

info "推送 $AHEAD 个提交到 $REMOTE/$BRANCH …"

if git push "$REMOTE" "$BRANCH"; then
    info "全部完成！分支: $BRANCH | 提交: $MSG"
else
    error "推送失败"
    if [[ "$REMOTE_URL" == https://* ]]; then
        echo "  HTTPS 方式需要配置个人访问令牌:"
        echo "  1. GitHub → Settings → Developer settings → Personal access tokens"
        echo "  2. 生成后设置远程地址:"
        echo "     git remote set-url origin https://<token>@github.com/Kevin-Cao96/neural-networks.git"
    fi
    exit 1
fi
