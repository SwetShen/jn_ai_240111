@echo off

echo 开始拉取
git pull
echo 拉取完成
echo 开始同步子仓库
git submodule update --remote
echo 同步子仓库完成

pause