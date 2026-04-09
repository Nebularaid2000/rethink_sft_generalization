import json

with open("/fs-computility/liudongrui/renqihan/cairuikun/harmbench2/test_cases/test_cases.json","r") as f:
    data = json.load(f)
    labels_with_length_not_1 = []
    for label, value in data.items():
        if isinstance(value, list) and len(value) != 1:
            labels_with_length_not_1.append(label)

    # 输出结果
    if labels_with_length_not_1:
        print(f"以下label的列表长度不等于1：{labels_with_length_not_1}")
    else:
        print("所有label的列表长度都等于1。")