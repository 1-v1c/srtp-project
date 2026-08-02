"""CalibTool 网页版启动器（工具包内，路径全自动定位）
双击 启动CalibTool网页版.bat 即可：
1. 自动定位到本工具包的 web/ 目录
2. 起本地 HTTP 服务器（浏览器安全策略要求，file:// 无法加载模型）
3. 自动打开浏览器
"""
import http.server
import os
import socketserver
import sys
import webbrowser

BASE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE, "web")


class Handler(http.server.SimpleHTTPRequestHandler):
    """带 MIME 修正 + /models/ 跨目录映射的静态文件处理器（.mjs/.wasm 必须正确标注）"""

    def translate_path(self, path):
        # 模型文件统一放在工具包根目录 models/（web/ 之外），
        # 页面请求 /models/face_landmarker.task 时映射到包根，避免 404
        if path.startswith("/models/"):
            mapped = os.path.normpath(os.path.join(BASE, path.lstrip("/")))
            models_root = os.path.join(BASE, "models")
            if mapped.startswith(models_root + os.sep):
                return mapped
        return super().translate_path(path)

    def guess_type(self, path):
        if path.endswith(".mjs"):
            return "text/javascript"
        if path.endswith(".wasm"):
            return "application/wasm"
        if path.endswith(".task"):
            return "application/octet-stream"
        return super().guess_type(path)

    def log_message(self, fmt, *args):
        pass  # 静默，避免终端刷屏


def main():
    if not os.path.exists(os.path.join(WEB_DIR, "index.html")):
        print("[ERROR] 未找到 web/index.html，工具包可能不完整")
        input("按回车退出...")
        sys.exit(1)

    os.chdir(WEB_DIR)

    # 尝试 8000~8009 端口，避免占用
    httpd = None
    for port in range(8000, 8010):
        try:
            httpd = socketserver.TCPServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if httpd is None:
        print("[ERROR] 端口 8000-8009 均被占用，请关闭占用程序后重试")
        input("按回车退出...")
        sys.exit(1)

    url = f"http://127.0.0.1:{port}"
    print("=" * 52)
    print("  CalibTool 网页版已启动")
    print(f"  地址: {url}")
    print("  保持本窗口开启，关闭窗口即退出")
    print("=" * 52)

    webbrowser.open(url)  # 自动打开浏览器

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
