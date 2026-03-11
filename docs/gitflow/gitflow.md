# GitFlow 开发常用命令速查表

本项目使用 **GitFlow 风格开发流程**：

```
main        稳定版本
develop     日常开发主分支
feature/*   功能开发
fix/*       bug修复
docs/*      文档更新
exp/*       实验代码
```

------

# 一、初始化仓库

初始化 Git：

```bash
git init
```

添加远程仓库：

```bash
git remote add origin https://github.com/username/repo.git
```

查看远程仓库：

```bash
git remote -v
```

第一次推送：

```bash
git push -u origin main
```

------

# 二、查看仓库状态

查看当前状态：

```bash
git status
```

查看提交历史：

```bash
git log
```

简洁历史：

```bash
git log --oneline --graph --decorate
```

查看当前分支：

```bash
git branch
```

查看所有分支：

```bash
git branch -a
```

------

# 三、分支操作

创建分支：

```bash
git branch feature/xxx
```

创建并切换分支（推荐）：

```bash
git switch -c feature/xxx
```

切换分支：

```bash
git switch develop
```

旧写法（仍可用）：

```bash
git checkout feature/xxx
```

删除分支：

```bash
git branch -d feature/xxx
```

强制删除：

```bash
git branch -D feature/xxx
```

------

# 四、日常开发流程（最常用）

### 1 开发前更新代码

```bash
git pull origin develop
```

------

### 2 新建功能分支

```bash
git switch develop
git pull origin develop
git switch -c feature/xxx
```

------

### 3 添加文件

添加全部文件：

```bash
git add .
```

添加指定文件：

```bash
git add path/to/file
```

------

### 4 提交代码

```bash
git commit -m "feat: add dataset visualization script"
```

------

### 5 推送代码

```bash
git push origin feature/xxx
```

第一次推送建议：

```bash
git push -u origin feature/xxx
```

------

# 五、更新分支（同步 develop）

开发过程中需要同步主分支：

```bash
git switch develop
git pull origin develop
git switch feature/xxx
git merge develop
```

如果出现冲突，解决后：

```bash
git add .
git commit
```

------

# 六、查看代码变化

查看修改内容：

```bash
git diff
```

查看某文件变化：

```bash
git diff file.py
```

查看已暂存变化：

```bash
git diff --cached
```

------

# 七、修改提交

修改最后一次 commit：

```bash
git commit --amend
```

修改作者信息：

```bash
git commit --amend --author="Name <email>"
```

------

# 八、撤销操作

取消暂存：

```bash
git restore --staged file.py
```

撤销文件修改：

```bash
git restore file.py
```

撤销所有修改：

```bash
git restore .
```

------

# 九、回退版本

回退一个 commit：

```bash
git reset --soft HEAD~1
```

回退并丢弃修改：

```bash
git reset --hard HEAD~1
```

------

# 十、远程仓库操作

拉取远程更新：

```bash
git pull origin develop
```

推送代码：

```bash
git push origin branch-name
```

强制推送（慎用）：

```bash
git push --force
```

------

# 十一、标签（用于版本发布）

创建 tag：

```bash
git tag v0.1
```

推送 tag：

```bash
git push origin v0.1
```

推送所有 tag：

```bash
git push origin --tags
```

------

# 十二、清理仓库

查看未追踪文件：

```bash
git clean -n
```

删除未追踪文件：

```bash
git clean -f
```

删除未追踪文件和目录：

```bash
git clean -fd
```

------

# 十三、推荐 Commit 规范

推荐使用 **Conventional Commit**：

```
feat: 新功能
fix: 修复bug
docs: 文档更新
refactor: 重构
test: 测试
chore: 其他修改
```

示例：

```
feat(data): add EMNIST dataset visualization script
fix(model): correct optical layer normalization
docs: update project structure
```

------

# 十四、常见开发流程示例

完整开发流程：

```bash
git switch develop
git pull origin develop

git switch -c feature/dataset-check

git add .
git commit -m "feat(scripts): add dataset sanity check script"

git push -u origin feature/dataset-check
```

------

# 十五、推荐分支命名

```
feature/dataset-check
feature/optical-model
feature/training-loop

fix/dataloader-bug

docs/gitflow
docs/experiment-log

exp/ablation-loss
exp/interpolation-test
```

git ls-files --others --exclude-standard