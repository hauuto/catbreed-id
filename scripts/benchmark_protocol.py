"""The handoff protocol: exactly 50 trials; exclude the first 5 from statistics."""


def collect_trials(items, predict):
    items = list(items)
    if len(items) != 50:
        raise ValueError("Benchmark requires exactly 50 input images")
    rows = []
    for index, item in enumerate(items):
        record = dict(predict(item))
        record.update(trial=index + 1, warmup=index < 5)
        rows.append(record)
    return rows
