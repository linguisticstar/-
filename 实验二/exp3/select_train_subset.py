import pandas as pd

# 原始训练集路径
input_path = "dataset/train.csv"
# 抽样结果保存路径
output_path = "dataset/train_2000.csv"

def main():
    # 加载数据（label,title,text）
    df = pd.read_csv(input_path, header=None, names=['label', 'title', 'text'])

    # 查看标签数量
    print("原始训练集标签统计：")
    print(df['label'].value_counts())

    # 统计每类数量
    count_bad = (df['label'] == 1).sum()   # 差评
    count_good = (df['label'] == 2).sum()  # 好评

    print(f"\n共有 差评(label=1)：{count_bad} 条")
    print(f"共有 好评(label=2)：{count_good} 条\n")

    # 实际抽取数量（如果不足1000，就全部取）
    n_bad = min(1000, count_bad)
    n_good = min(1000, count_good)

    print(f"将抽取 差评(label=1)：{n_bad} 条")
    print(f"将抽取 好评(label=2)：{n_good} 条\n")

    # 抽样
    df_bad = df[df['label'] == 1].sample(n=n_bad, random_state=42)
    df_good = df[df['label'] == 2].sample(n=n_good, random_state=42)

    # 合并 & 打乱
    df_new = pd.concat([df_bad, df_good]).sample(frac=1, random_state=42)

    # 保存，不带 header 和 index
    df_new.to_csv(output_path, index=False, header=False, encoding="utf-8")

    print("🎉 新训练集生成成功！")
    print("保存路径：", output_path)
    print("最终数据条数：", len(df_new))

if __name__ == "__main__":
    main()
