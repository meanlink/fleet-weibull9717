import hashlib, json
from src.models import Dataset
from src.services.business import legacy_samples


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def dataset_dict(d):
    return {'id':d.id, 'name':d.name, 'source':d.source, 'rows':len(d.records), 'checksum':d.checksum, 'created_at':d.created_at}


# 结果在任务完成时落库；页面、批准和分发下载都读取同一份快照。
# 仅当结果来自本功能上线前且该批次每件都从0公里起算时，才补逐件起止里程。
def _job_result(j, db):
    result = j.result or {}
    if 'samples' in result or j.status != 'completed':
        return result
    dataset = db.get(Dataset, j.dataset_id)
    if not dataset:
        return result
    samples = legacy_samples(dataset.records, str((j.params or {}).get('model', 'AC-8')))
    return {**result, 'samples': samples} if samples else result


def job_result(j, db):
    return _job_result(j, db)


def job_dict(j, db=None):
    data = {k: getattr(j, k) for k in ['id', 'name', 'dataset_id', 'params', 'status', 'error', 'approved', 'created_at', 'finished_at']}
    data['result'] = _job_result(j, db) if db is not None else (j.result or {})
    return data


def report_dict(r):
    return {k: getattr(r, k) for k in ['id', 'name', 'job_id', 'destination', 'interval_seconds', 'next_run', 'enabled']}


def delivery_dict(d):
    return {k: getattr(d, k) for k in ['id', 'report_id', 'created_at', 'checksum', 'filename', 'destination']}
