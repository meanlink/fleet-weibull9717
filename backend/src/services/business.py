"""Two-parameter Weibull maximum likelihood with right-censoring and left-truncation."""
import math
import random
import numpy as np
from scipy.optimize import minimize


def metadata():
    return {
        "name": "车队部件寿命估计", "short": "fleet-weibull", "tagline": "保留尚未失效部件，用删失似然估计预防检修窗口",
        "client": "模拟客户：鹭湾公交机务组，96 辆公交车跟踪两种空气压缩机。",
        "workflow": "机务组每月导出更换与在役记录，只对已坏件求平均寿命，再用统一里程安排检修。",
        "pains": ["只看坏件会丢掉在役件的寿命信息", "不同压缩机型号被混在一个均值内", "检修里程未表达可靠度假设"],
        "upstream": "部件安装里程、失效更换里程与截止日仍在役里程", "downstream": "机务检修评审，人工结合安全规程安排检查。",
        "entities": ["部件寿命样本", "可靠度估计", "机务简报计划"], "dataset_name": "鹭湾公交_压缩机寿命与在役截止.csv",
        "columns": [{"key": "asset", "label": "部件资产号", "type": "string", "required": True}, {"key": "model", "label": "型号", "type": "string", "required": True}, {"key": "km", "label": "观测运行里程", "type": "number", "required": True}, {"key": "failed", "label": "已失效1/在役0", "type": "number", "required": True}, {"key": "start_km", "label": "开始跟踪里程（可选）", "type": "number", "required": False, "description": "部件进入车队跟踪时已运行的里程；留空或缺失表示从 0 公里开始跟踪。"}],
        "parameters": [{"key": "model", "label": "部件型号", "type": "select", "default": "AC-8", "options": [{"value": "AC-8", "label": "AC-8 螺杆机"}, {"value": "AC-12", "label": "AC-12 改进型"}]}, {"key": "reliability", "label": "目标可靠度", "type": "number", "default": 0.9, "min": 0.6, "max": 0.99}],
        "workbench_title": "删失样本与检修可靠度", "action_label": "拟合寿命曲线", "unit": "公里", "accent": "oklch(51% 0.13 35)", "style": "fleet-chart",
        "seed_story": ["08:00 导入鹭湾公交_压缩机寿命与在役截止.csv，共200件。", "08:20 对照资产编号确认 failed=0 是截止仍在役，不是零寿命。", "08:25 新接管部件填写开始跟踪里程；未填写的旧记录按从0公里跟踪。", "09:00 选择 AC-8，目标可靠度设为0.90。", "09:15 执行含右删失和左截断的Weibull极大似然估计。", "09:30 查看形状参数、B10寿命和目标可靠度里程。", "10:00 切到AC-12独立拟合，避免混合型号。", "14:00 将曲线与样本结构提交机务评审，并创建本地分发记录。"],
        "method": "独立右删失且条件于已存活至开始跟踪里程的条件下，L=∏[f(t)/S(t0)]^δ[S(t)/S(t0)]^(1−δ)，t0为开始跟踪里程（缺省0），f为二参数Weibull密度，S(t)=exp[−(t/η)^β]。在logβ/logη空间求极大似然；目标里程t=η[−ln R]^(1/β)。",
        "limitations": ["假设独立删失和同质运行工况，未纳入线路、载荷、批次等协变量。", "里程阈值是样本模型估计，不替代厂家安全规范，也未包含参数置信区间。"],
    }


def seed_records():
    rng = random.Random(1017)
    rows = []
    for model, scale, shape in (("AC-8", 88000, 2.4), ("AC-12", 116000, 1.8)):
        for index in range(100):
            failure = rng.weibullvariate(scale, shape)
            cutoff = rng.uniform(35000, 145000)
            rows.append({"asset": f"{model}-{index+1:03}", "model": model, "km": round(min(failure, cutoff), 1), "failed": int(failure <= cutoff)})
    # 新接管部件在运行一段里程后才进入跟踪；独立随机序列，不改动上面的既有样本。
    tracking = random.Random(2017)
    for row in rows:
        if tracking.random() < 0.25:
            row["start_km"] = round(min(tracking.uniform(20000, 60000), row["km"] * 0.8), 1)
    return rows


START_KEY = "start_km"
MAX_KM = 10_000_000


def _start_mileage(row, time, asset):
    value = row.get(START_KEY)
    if value is None or (isinstance(value, str) and not value.strip()):
        return 0.0, "legacy"
    try:
        if isinstance(value, bool):
            raise ValueError()
        start = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"资产 {asset} 的开始跟踪里程必须是有限数值") from None
    if not math.isfinite(start) or not 0 <= start <= MAX_KM:
        raise ValueError("开始跟踪里程须为0至1000万公里的有限数值")
    if start >= time:
        raise ValueError(f"资产 {asset} 的开始跟踪里程不早于观测里程，无法估计")
    return start, "recorded"


def _model_rows(records, model_name, relation_only=False):
    """Parse one model's rows and apply the per-row start-mileage rule.

    Old batches have no start_km column at all (or a blank cell); such rows start
    tracking at 0 km. Mixed batches are judged row by row.
    """
    samples = []
    assets = set()
    for row in records:
        if not isinstance(row, dict) or str(row.get("model")) != model_name:
            continue
        try:
            time = float(row["km"])
            failed = float(row["failed"])
            asset = str(row["asset"])
        except (KeyError, ValueError, TypeError):
            if relation_only:
                continue
            raise ValueError("每条记录须有资产号、正里程和0/1失效标记") from None
        if not math.isfinite(time) or not 1 <= time <= MAX_KM or failed not in (0.0, 1.0):
            if relation_only:
                continue
            raise ValueError("里程须为1至1000万公里的有限数值，failed只能是0或1")
        if not asset or (not relation_only and asset in assets):
            if relation_only:
                continue
            raise ValueError("资产号为空或重复，不能重复计算同一部件")
        assets.add(asset)
        start, source = _start_mileage(row, time, asset)
        samples.append({"asset": asset, "model": model_name, "start_km": start, "km": time, "failed": int(failed), "start_source": source})
    return samples


def _sample_rows(samples):
    return [{"asset": sample["asset"], "model": sample["model"], "start_km": round(sample["start_km"], 1), "km": round(sample["km"], 1), "end_km": round(sample["km"], 1), "failed": sample["failed"], "start_source": sample["start_source"]} for sample in samples]


SAMPLE_COLUMNS = [{"key": "asset", "label": "部件资产号"}, {"key": "start_km", "label": "开始跟踪里程"}, {"key": "km", "label": "观测里程（截止/失效）"}, {"key": "failed", "label": "已失效1/在役0"}, {"key": "start_source", "label": "起算口径"}]


def legacy_samples(records, model_name):
    """Per-part start/end mileage for results computed before this feature.

    Only safe when every selected row was in fact fitted from 0 km.
    """
    selected = [row for row in records if isinstance(row, dict) and str(row.get("model")) == model_name]
    recorded = [row for row in selected if row.get("start_km") not in (None, "", "  ")]
    if not selected:
        return None
    for row in recorded:
        try:
            start, _ = _start_mileage(row, MAX_KM + 1, str(row.get("asset", "")))
        except ValueError:
            return None
        if start != 0:
            return None
    try:
        samples = _model_rows(records, model_name, relation_only=True)
    except ValueError:
        return None
    return {"columns": SAMPLE_COLUMNS, "rows": _sample_rows(samples)} if samples else None


def validate_start_relation(records, model_name):
    """Early guard used when submitting a job; the full check runs at estimation."""
    _model_rows(records, model_name, relation_only=True)


def analyze(records, params):
    if not isinstance(records, list) or not records:
        raise ValueError("请先提供含失效标记的部件寿命样本")
    model_name = str(params.get("model", "AC-8"))
    try:
        reliability = float(params.get("reliability", 0.9))
    except (ValueError, TypeError):
        raise ValueError("目标可靠度必须是数值") from None
    if not math.isfinite(reliability) or not 0.6 <= reliability <= 0.99:
        raise ValueError("目标可靠度须介于0.6和0.99")
    samples = _model_rows(records, model_name)
    times = [sample["km"] for sample in samples]
    events = [sample["failed"] for sample in samples]
    starts = np.asarray([sample["start_km"] for sample in samples], dtype=float)
    truncated = sum(1 for sample in samples if sample["start_source"] == "recorded" and sample["start_km"] > 0)
    failures = sum(events)
    if len(times) < 25 or failures < 10:
        raise ValueError("所选型号至少需要25件样本和10件已失效样本")
    t = np.asarray(times, dtype=float)
    event = np.asarray(events, dtype=float)
    log_t = np.log(t)
    log_start = np.log(np.where(starts > 0, starts, 1.0))

    def objective(theta):
        shape, scale = np.exp(theta)
        powers = np.exp(np.clip(shape * (log_t - math.log(scale)), -600, 600))
        # 条件于部件已存活到开始跟踪里程：减去各件在 t0 处的累积风险。
        powers_start = np.where(starts > 0, np.exp(np.clip(shape * (log_start - math.log(scale)), -600, 600)), 0.0)
        log_density = math.log(shape) - shape * math.log(scale) + (shape - 1) * log_t
        return float(np.sum(powers - powers_start - event * log_density))

    solution = minimize(objective, np.log([1.8, float(np.median(t)) * 1.3]), method="L-BFGS-B", bounds=[(math.log(0.1), math.log(15)), (math.log(float(t.min()) * 0.1), math.log(float(t.max()) * 100))])
    if not solution.success or not math.isfinite(float(solution.fun)):
        raise ValueError("寿命似然未收敛，请检查样本或增加失效观测")
    shape, scale = [float(value) for value in np.exp(solution.x)]
    b10 = scale * (-math.log(0.9)) ** (1 / shape)
    threshold = scale * (-math.log(reliability)) ** (1 / shape)
    mean_life = scale * math.gamma(1 + 1 / shape)
    levels = sorted({0.99, 0.95, 0.9, 0.8, 0.7, reliability}, reverse=True)
    rows = [{"reliability": round(level, 4), "km": round(scale * (-math.log(level)) ** (1 / shape), 1), "failed_pct": round((1-level)*100, 2), "target": "本次目标" if level == reliability else "参考水平"} for level in levels]
    chart_x = np.linspace(0, float(t.max()) * 1.2, 41)
    curve = [round(math.exp(-(float(value) / scale) ** shape) * 100, 4) for value in chart_x]
    censored = len(times) - failures
    legacy = len(samples) - truncated
    warnings = []
    if censored / len(times) > 0.6:
        warnings.append("超过60%的样本仍在役，尾部寿命外推不稳定。")
    if threshold > max(times):
        warnings.append("目标可靠度里程超过最大观测里程，属于外推。")
    if shape < 0.11 or shape > 14.9:
        warnings.append("形状参数接近优化边界，当前二参数假设可能不适合数据。")
    tracking_note = f"其中{truncated}件按开始跟踪里程条件拟合" + (f"，{legacy}件旧数据按0公里起算" if legacy else "") + "；" if truncated else ""
    sample_rows = _sample_rows(samples)
    return {
        "summary": f"{model_name}的{len(times)}件样本含{failures}件失效、{censored}件右删失；{tracking_note}目标可靠度{reliability:.0%}对应运行里程约{threshold:,.0f}公里。",
        "metrics": [{"label": "形状参数β", "value": round(shape, 4)}, {"label": "尺度参数η", "value": round(scale, 1), "unit": "公里"}, {"label": "B10寿命", "value": round(b10, 1), "unit": "公里"}, {"label": "目标可靠度里程", "value": round(threshold, 1), "unit": "公里"}],
        "table": {"columns": [{"key": "reliability", "label": "可靠度R"}, {"key": "km", "label": "运行里程"}, {"key": "failed_pct", "label": "累计失效概率%"}, {"key": "target", "label": "用途"}], "rows": rows},
        "samples": {"columns": SAMPLE_COLUMNS, "rows": sample_rows},
        "chart": {"type": "line", "labels": [str(round(float(value))) for value in chart_x], "series": [{"name": "仍正常运行概率%", "values": curve}]},
        "findings": [f"β={shape:.3f}，{'提示随里程增加的磨损趋势' if shape > 1 else '提示早期失效或近似恒定风险，应检查样本同质性'}；这不是因果判断。", f"模型平均寿命为{mean_life:,.0f}公里；均值不等于预防检修推荐里程。", f"在役件贡献生存概率，不被错误当成已失效件；删失比例{censored/len(times):.1%}。" + (f" 新接管部件按已存活到开始跟踪里程的条件进入似然，避免把其已有里程当作零寿命。共{truncated}件记录了开始跟踪里程。" if truncated else "")],
        "method": metadata()["method"], "warnings": warnings,
    }
