"""CalibTool 桌面版启动器（工具包内，路径全自动定位）
双击 启动CalibTool桌面版.bat 即可。

为什么需要这个启动器：
- MediaPipe 底层 C 库不支持中文路径，若工具包位于含中文的目录，
  模型必须先复制到纯英文临时目录再加载（通过环境变量覆盖）。
"""
import os
import shutil
import subprocess
import sys
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE, "desktop", "src")
MODEL_SRC = os.path.join(BASE, "models", "face_landmarker.task")


def prepare_model():
    """把模型复制到纯英文临时路径，返回模型路径"""
    tmp_dir = os.path.join(tempfile.gettempdir(), "calibtool_models")
    os.makedirs(tmp_dir, exist_ok=True)
    dst = os.path.join(tmp_dir, "face_landmarker.task")

    # 若已存在且大小一致，跳过复制
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(MODEL_SRC):
        return dst

    shutil.copy2(MODEL_SRC, dst)
    return dst


def main():
    if not os.path.exists(os.path.join(SRC_DIR, "calibtool.py")):
        print("[ERROR] 未找到 desktop/src/calibtool.py，工具包可能不完整")
        input("按回车退出...")
        sys.exit(1)
    if not os.path.exists(MODEL_SRC):
        print("[ERROR] 未找到 models/face_landmarker.task，工具包可能不完整")
        input("按回车退出...")
        sys.exit(1)

    # 模型路径走纯英文临时目录
    os.environ["CALIBTOOL_MODEL_PATH"] = prepare_model()

    print("=" * 52)
    print("  CalibTool 桌面版启动中...")
    print("  窗口出现后即可开始标定（ESC 或 Q 退出）")
    print("=" * 52)

    try:
        subprocess.run([sys.executable, "calibtool.py"], cwd=SRC_DIR)
    except KeyboardInterrupt:
        pass

    print("\nCalibTool 已退出。")


if __name__ == "__main__":
    main()
