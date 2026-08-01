"""
核心逻辑单元测试（模块 09 部分 — 不依赖摄像头/模型）
运行：python tests/test_core.py
覆盖：FatigueScorer 评分/等级/衰减/重置 + PERCLOSCounter
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fatigue import FatigueScorer
from features import PERCLOSCounter, BlinkAnalyzer


def approx(a, b, tol=0.01):
    return abs(a - b) <= tol


def test_scorer_idle():
    """无任何事件 → 分数 0，等级 L0"""
    s = FatigueScorer()
    for _ in range(30):
        s.update(1 / 30, {"eye_closed_sec": 0.0})
    assert approx(s.get_score(), 0.0), f"idle score={s.get_score()}"
    lvl, label = s.get_level()
    assert lvl == 0 and "L0" in label, f"idle level={label}"
    print("[OK] test_scorer_idle")


def test_scorer_yawn_event():
    """一次哈欠 → yawn 源 +2，总分数 ≈2 → L1"""
    s = FatigueScorer()
    s.update(1 / 30, {"yawn_just_completed": True, "eye_closed_sec": 0.0})
    assert approx(s.get_score(), 2.0, 0.1), f"score={s.get_score()}"
    lvl, label = s.get_level()
    assert lvl == 1, f"level={label}"
    srcs = dict(s.get_active_sources())
    assert "yawn" in srcs, f"sources={srcs}"
    print("[OK] test_scorer_yawn_event")


def test_scorer_eye_close_sustain():
    """持续闭眼 1 秒：前 0.5s 为眨眼过滤期（不计分），后 0.5s 按 4分/s 累积 ≈2 → L1 边界"""
    s = FatigueScorer()
    fps = 30
    for i in range(fps):  # 闭眼 1 秒
        s.update(1 / fps, {"eye_closed_sec": (i + 1) / fps})
    score = s.get_score()
    assert 1.8 <= score <= 2.3, f"score={score}"
    lvl, _ = s.get_level()
    assert lvl == 1, f"level={lvl}"
    print(f"[OK] test_scorer_eye_close_sustain (score={score:.2f})")


def test_scorer_micro_sleep():
    """闭眼跨过 2s：eye_close 累积(1.5s×4=6) + 微睡眠事件 +4 ≈10 → L3"""
    s = FatigueScorer()
    fps = 30
    for i in range(int(2.5 * fps)):
        s.update(1 / fps, {"eye_closed_sec": (i + 1) / fps})
    score = s.get_score()
    srcs = dict(s.get_active_sources())
    assert "micro_sleep" in srcs, f"sources={srcs}"
    assert 9.0 <= score <= 11.0, f"score={score}"
    lvl, _ = s.get_level()
    assert lvl == 3, f"level={lvl}"
    print(f"[OK] test_scorer_micro_sleep (score={score:.2f}, level={lvl})")


def test_scorer_head_down():
    """持续低头 2 秒 → head_down 按 1分/s 累积 ≈2"""
    s = FatigueScorer()
    fps = 30
    for i in range(int(2.0 * fps)):
        s.update(1 / fps, {"head_down": True, "eye_closed_sec": 0.0})
    srcs = dict(s.get_active_sources())
    assert "head_down" in srcs, f"sources={srcs}"
    assert 1.5 <= s.get_score() <= 2.5, f"score={s.get_score()}"
    print("[OK] test_scorer_head_down")


def test_scorer_decay():
    """事件停止后分数按半衰期回落（45s 半衰期，10s 后约剩 86%）"""
    s = FatigueScorer()
    s.update(0.1, {"yawn_just_completed": True, "eye_closed_sec": 0.0})
    s0 = s.get_score()
    assert approx(s0, 2.0, 0.1)
    for _ in range(100):  # 再跑 10 秒无事件
        s.update(0.1, {"eye_closed_sec": 0.0})
    s1 = s.get_score()
    ratio = s1 / s0
    assert 0.80 <= ratio <= 0.90, f"decay ratio={ratio:.3f} (期望≈0.857)"
    print(f"[OK] test_scorer_decay (10s 后剩 {ratio*100:.1f}%)")


def test_scorer_level_boundary():
    """等级边界：2→L1, 5→L2, 9→L3"""
    s = FatigueScorer()
    # 用事件源直接构造分数
    s._sources["yawn"]["score"] = 1.9
    assert s.get_level()[0] == 0
    s._sources["yawn"]["score"] = 2.0
    assert s.get_level()[0] == 1
    s._sources["yawn"]["score"] = 5.0
    assert s.get_level()[0] == 2
    s._sources["yawn"]["score"] = 9.0
    assert s.get_level()[0] == 3
    print("[OK] test_scorer_level_boundary")


def test_scorer_reset():
    """reset 后全部归零"""
    s = FatigueScorer()
    s.update(0.1, {"yawn_just_completed": True, "eye_closed_sec": 2.5})
    assert s.get_score() > 0
    s.reset()
    assert approx(s.get_score(), 0.0)
    assert s.get_active_sources() == []
    print("[OK] test_scorer_reset")


def test_perclos_counter():
    """PERCLOS：窗口 2s@10fps，前 10 帧闭眼（1s>0.3s 豁免）后 10 帧睁眼 → 0.50"""
    c = PERCLOSCounter(window_sec=2, ear_threshold=0.22, fps=10)
    for _ in range(10):
        c.update(0.10)   # 闭眼 1s，超过豁免期 → 全部计入
    for _ in range(10):
        c.update(0.35)   # 睁眼
    assert approx(c.get_perclos(), 0.50, 0.02), f"perclos={c.get_perclos()}"
    # 无人脸帧跳过
    c.update(None)
    assert approx(c.get_perclos(), 0.50, 0.02)
    # 窗口不满按已有数据算（5 帧闭眼 = 0.5s > 豁免期）
    c2 = PERCLOSCounter(window_sec=10, ear_threshold=0.22, fps=10)
    for _ in range(5):
        c2.update(0.10)
    assert approx(c2.get_perclos(), 1.0)
    print("[OK] test_perclos_counter")


def test_perclos_blink_exemption():
    """眨眼豁免：单次眨眼（2帧=0.2s < 0.3s）不计入闭眼；长闭眼计入"""
    c = PERCLOSCounter(window_sec=10, ear_threshold=0.22, fps=10)
    # 眨眼：闭 2 帧 + 睁 2 帧（0.2s < 0.3s 豁免）
    for _ in range(2):
        c.update(0.10)
    for _ in range(2):
        c.update(0.35)
    assert approx(c.get_perclos(), 0.0, 0.01), f"眨眼应豁免 perclos={c.get_perclos()}"
    # 长闭眼：4 帧 = 0.4s > 0.3s 豁免 → 计入
    for _ in range(4):
        c.update(0.10)
    for _ in range(2):
        c.update(0.35)
    p = c.get_perclos()
    assert approx(p, 4.0 / 10.0, 0.02), f"perclos={p}"
    # 豁免关闭时（blink_exempt_sec=0）眨眼也计入
    c3 = PERCLOSCounter(window_sec=10, ear_threshold=0.22, fps=10, blink_exempt_sec=0.0)
    for _ in range(2):
        c3.update(0.10)
    for _ in range(2):
        c3.update(0.35)
    assert approx(c3.get_perclos(), 0.5, 0.02), f"perclos={c3.get_perclos()}"
    print("[OK] test_perclos_blink_exemption")


def test_blink_analyzer():
    """眨眼分析器：正常眨眼（0.2s）计入统计；闭合速度为正"""
    fps = 30
    dt = 1.0 / fps
    t = 100.0
    a = BlinkAnalyzer(ear_threshold=0.22, window_sec=60.0,
                      slow_blink_sec=0.5, long_closure_sec=2.0)

    # 正常睁开基线 0.4
    for _ in range(10):
        a.update(0.40, t)
        t += dt

    # 一次正常眨眼：EAR 从 0.40 跌到 0.08 再回升，总时长 6 帧 = 0.2s
    blink_frames = 6
    for i in range(blink_frames):
        a.update(0.40 - (0.40 - 0.08) * ((i + 1) / blink_frames), t)  # 下降
        t += dt
    for i in range(blink_frames):
        a.update(0.08 + (0.40 - 0.08) * ((i + 1) / blink_frames), t)  # 回升
        t += dt

    stats = a.get_stats(now=t)
    # 窗口 60s 内只眨了 1 次 → 1 次/分
    assert approx(stats["blink_rate"], 1.0, 0.1), f"rate={stats['blink_rate']}"
    assert 0.15 <= stats["avg_duration"] <= 0.25, f"dur={stats['avg_duration']}"
    assert stats["avg_closing_speed"] > 0.5, f"speed={stats['avg_closing_speed']}"
    print(f"[OK] test_blink_analyzer (rate={stats['blink_rate']:.0f}/min, dur={stats['avg_duration']*1000:.0f}ms, speed={stats['avg_closing_speed']:.2f})")


def test_blink_analyzer_slow_blink():
    """慢眨眼（0.6s）计入统计且标记为 slow；长闭眼（3s）不计为眨眼"""
    fps = 30
    dt = 1.0 / fps
    t = 100.0
    a = BlinkAnalyzer(ear_threshold=0.22, window_sec=60.0,
                      slow_blink_sec=0.5, long_closure_sec=2.0)
    for _ in range(10):
        a.update(0.40, t)
        t += dt

    # 慢眨眼：闭 18 帧 = 0.6s（>0.5s 慢眨眼，<2s 计入）
    for _ in range(18):
        a.update(0.08, t)
        t += dt
    for _ in range(5):
        a.update(0.40, t)
        t += dt
    stats = a.get_stats(now=t)
    assert approx(stats["blink_rate"], 1.0, 0.1), f"rate={stats['blink_rate']}"
    assert approx(stats["slow_blink_rate"], 1.0, 0.1), f"slow={stats['slow_blink_rate']}"
    assert 0.5 <= stats["avg_duration"] <= 0.7, f"dur={stats['avg_duration']}"

    # 长闭眼 3s：90 帧 ≥ 2s → 不计为眨眼
    for _ in range(90):
        a.update(0.08, t)
        t += dt
    for _ in range(5):
        a.update(0.40, t)
        t += dt
    stats = a.get_stats(now=t)
    assert approx(stats["blink_rate"], 1.0, 0.1), f"rate={stats['blink_rate']}"
    print("[OK] test_blink_analyzer_slow_blink")


def test_blink_analyzer_long_closure():
    """无人脸重置后统计清零"""
    fps = 30
    dt = 1.0 / fps
    t = 100.0
    a = BlinkAnalyzer(ear_threshold=0.22, window_sec=60.0,
                      slow_blink_sec=0.5, long_closure_sec=2.0)
    for _ in range(10):
        a.update(0.40, t)
        t += dt
    for _ in range(6):
        a.update(0.08, t)
        t += dt
    for _ in range(5):
        a.update(0.40, t)
        t += dt
    assert a.get_stats(now=t)["blink_rate"] > 0
    # 无人脸只重置闭合状态，不清空滑动窗口统计
    a.update(None, t)
    assert a.get_stats(now=t)["blink_rate"] > 0
    # reset() 清空全部统计
    a.reset()
    assert a.get_stats(now=t)["blink_rate"] == 0.0
    print("[OK] test_blink_analyzer_long_closure")


def test_scorer_blink_sources():
    """眨眼疲劳源：频率/时长异常时按速率累积分数（300s 达稳态）"""
    s = FatigueScorer()
    fps = 30
    # 模拟 300 秒眨眼异常状态（稳态：rate/λ，λ=ln2/45）
    for i in range(300 * fps):
        s.update(1 / fps, {
            "eye_closed_sec": 0.0,
            "blink_rate_over": True,
            "blink_dur_over": True,
        })
    srcs = dict(s.get_active_sources())
    assert "blink_rate" in srcs, f"sources={srcs}"
    assert "blink_dur" in srcs, f"sources={srcs}"
    # 稳态值：rate × half_life / ln2 ≈ 1.08（rate=1/60）/ 2.16（rate=2/60）
    assert 0.9 <= srcs["blink_rate"] <= 1.3, f"blink_rate={srcs['blink_rate']}"
    assert 1.8 <= srcs["blink_dur"] <= 2.5, f"blink_dur={srcs['blink_dur']}"
    # 总分应 ≥ 2 → L1
    assert s.get_score() >= 2.0, f"score={s.get_score()}"
    print(f"[OK] test_scorer_blink_sources (score={s.get_score():.2f})")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n全部 {len(tests)} 项测试通过 [PASS]")
