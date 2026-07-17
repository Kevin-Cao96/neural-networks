#!/bin/zsh
# Mac GitHub 同步更新封装（含变基合并）
# 使用：chmod +x git_sync.sh
# ./git_sync.sh "更新备注"
# ./git_sync.sh 自动生成时间备注

# 配置
BRANCH="main"
REMOTE="origin"

# 拼接提交信息
if [[ $# -ge 1 ]]; then
  COMMIT_MSG="$1"
else
  COMMIT_MSG="Auto sync $(date '+%Y-%m-%d %H:%M:%S')"
fi

# 1. 判断是否Git仓库
if [[ ! -d ".git" ]]; then
  echo "\033[31m[错误] 当前目录不是Git仓库\033[0m"
  exit 1
fi

echo "==================== 1. 拉取远端并变基合并 ===================="
# git pull = fetch + merge，改用 --rebase 变基合并，历史更干净
git pull $REMOTE $BRANCH --rebase
PULL_RET=$?

if [[ $PULL_RET -eq 1 ]]; then
  echo "\033[31m[错误] 合并发生代码冲突，请手动解决冲突后再执行脚本\033[0m"
  exit 1
elif [[ $PULL_RET -ge 2 ]]; then
  echo "\033[31m[错误] 拉取远端失败，网络/远端分支不存在\033[0m"
  exit 1
fi

echo "==================== 2. 暂存所有修改 ===================="
git add .

# 判断是否存在待提交变更
git diff --cached --quiet
if [[ $? -eq 0 ]]; then
  echo "\033[32m[提示] 无本地文件修改，同步完成\033[0m"
  exit 0
fi

echo "==================== 3. 本地提交 ===================="
git commit -m "$COMMIT_MSG"

echo "==================== 4. 推送至GitHub远端 ===================="
git push $REMOTE $BRANCH
PUSH_RET=$?

if [[ $PUSH_RET -eq 0 ]]; then
  echo -e "\033[32m✅ 同步完成 | 分支:$BRANCH | 备注:$COMMIT_MSG\033[0m"
else
  echo "\033[31m[错误] 推送失败，远端有新提交，执行脚本重新拉取合并\033[0m"
  exit 1
fi
