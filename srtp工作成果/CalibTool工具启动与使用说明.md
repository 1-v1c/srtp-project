---
title: CalibTool 启动与使用说明
date: 2026-07-12
tags: [srtp, calibtool, 工具]
---

## CalibTool 启动方式

### 方案一：双击 .bat（推荐）
```
E:\ObsidianVault\.codex\启动CalibTool.bat
```

### 方案二：PowerShell 命令行
```powershell
cd "E:\ObsidianVault\01-Projects\SRTP项目—人车协同安全系统\fatigue_detection\src"
python calibtool.py
```

### 方案三：桌面快捷方式
目标位置填：
```
D:\Anaconda3\local\python.exe "E:\ObsidianVault\.codex\launch_calibtool.py"
```

---

## 快捷键

| 按键 | 功能 |
|:---|:------|
| **空格** | 开始 / 停止录制 |
| **Tab** | 前进到下一个标签 |
| **B** | 回退到上一个标签 |
| **1/2/3** | 切换阶段（EAR / MAR / HEAD） |
| **E** | 导出 CSV 数据 |
| **Backspace** | 删除选中记录（按两次确认） |
| **Q** | 退出（按两次确认，自动导出） |
| **H** | 查看帮助 |
| **鼠标滚轮** | 侧面板记录列表滚动 |

---

## 安装环境（队友首次使用）
```
E:\ObsidianVault\.codex\安装环境.bat
```
