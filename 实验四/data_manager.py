# data_manager.py - 数据管理工具
import os
import json
import re
import sys
import shutil
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f"📊 {title}")
    print("=" * 60)


def check_data_directory():
    """检查数据目录"""
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"创建数据目录: {data_dir}")

    html_files = [f for f in os.listdir(data_dir) if f.endswith('.html')]
    json_file = os.path.join(data_dir, "processed_data.json")

    return html_files, json_file


def generate_sample_html():
    """生成示例HTML文件"""
    sample_diseases = [
        {
            "name": "高血压",
            "symptoms": "头痛、头晕、心悸、胸闷、疲劳",
            "causes": "遗传因素、高盐饮食、肥胖、缺乏运动、压力过大",
            "treatment": "药物治疗（利尿剂、β受体阻滞剂）、低盐饮食、定期运动、控制体重"
        },
        {
            "name": "糖尿病",
            "symptoms": "多饮、多尿、多食、体重下降、视力模糊",
            "causes": "胰岛素分泌不足、胰岛素抵抗、遗传因素、肥胖",
            "treatment": "胰岛素注射、口服降糖药、饮食控制、运动疗法"
        },
        {
            "name": "冠心病",
            "symptoms": "胸痛、胸闷、心悸、气短、疲劳",
            "causes": "动脉粥样硬化、高血压、高血脂、吸烟、糖尿病",
            "treatment": "药物治疗（阿司匹林、他汀类）、冠状动脉介入治疗、搭桥手术"
        },
        {
            "name": "肺炎",
            "symptoms": "发热、咳嗽、咳痰、胸痛、呼吸困难",
            "causes": "细菌感染、病毒感染、真菌感染、吸入异物",
            "treatment": "抗生素治疗、抗病毒药物、止咳化痰药、氧疗"
        },
        {
            "name": "胃炎",
            "symptoms": "上腹痛、腹胀、恶心、呕吐、食欲不振",
            "causes": "幽门螺杆菌感染、药物刺激、饮食不当、压力过大",
            "treatment": "抗生素治疗、胃酸抑制剂、保护胃黏膜药物、饮食调整"
        }
    ]

    data_dir = "./data"
    created_files = []

    for disease in sample_diseases:
        filename = f"{disease['name']} - 医学百科.html"
        filepath = os.path.join(data_dir, filename)

        if not os.path.exists(filepath):
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>{disease['name']} - 医学百科</title>
            </head>
            <body>
                <h1>{disease['name']}</h1>

                <div class="content">
                    <h2>概述</h2>
                    <p>{disease['name']}是一种常见的疾病，需要及时诊断和治疗。</p>

                    <h2>症状</h2>
                    <p>{disease['symptoms']}</p>

                    <h2>病因</h2>
                    <p>{disease['causes']}</p>

                    <h2>治疗方法</h2>
                    <p>{disease['treatment']}</p>

                    <h2>预防措施</h2>
                    <p>1. 健康饮食<br>2. 规律运动<br>3. 定期体检<br>4. 避免危险因素</p>

                    <h2>注意事项</h2>
                    <p>如出现相关症状，请及时就医。本信息仅供参考，不能替代专业医疗建议。</p>
                </div>
            </body>
            </html>
            """

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            created_files.append(filename)
            print(f"✅ 创建: {filename}")

    return created_files


def extract_from_html(html_filepath):
    """从HTML提取内容"""
    try:
        with open(html_filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # 提取标题
        title_tag = soup.find('title')
        title = title_tag.text.strip() if title_tag else Path(html_filepath).stem

        # 提取正文
        content_tag = soup.find('div', class_='content')
        if not content_tag:
            content_tag = soup.find('body')

        if content_tag:
            text = content_tag.get_text(separator='\n', strip=True)
            # 清理文本
            text = re.sub(r'\n\s*\n', '\n', text)
            text = re.sub(r'\s+', ' ', text)
            return title, text
        else:
            return title, ""

    except Exception as e:
        print(f"❌ 提取失败 {html_filepath}: {e}")
        return None, None


def split_text(text, chunk_size=500, chunk_overlap=50):
    """分割文本"""
    if not text or len(text) < 50:
        return []

    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for space

        if current_length + word_length > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))

            # 保留重叠部分
            overlap_words = current_chunk[-min(len(current_chunk), chunk_overlap // 5):]
            current_chunk = overlap_words.copy()
            current_length = sum(len(w) + 1 for w in current_chunk)

        current_chunk.append(word)
        current_length += word_length

    # 添加最后一个块
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def process_html_files():
    """处理所有HTML文件"""
    data_dir = "./data"
    html_files = [f for f in os.listdir(data_dir) if f.endswith('.html')]

    if not html_files:
        print("❌ 没有找到HTML文件")
        return []

    all_data = []
    total_chunks = 0

    print_header("处理HTML文件")
    print(f"找到 {len(html_files)} 个HTML文件")

    for i, filename in enumerate(html_files, 1):
        filepath = os.path.join(data_dir, filename)
        print(f"\n[{i}/{len(html_files)}] 处理: {filename}")

        title, text = extract_from_html(filepath)

        if text and len(text) > 100:
            chunks = split_text(text)

            for j, chunk in enumerate(chunks):
                data_entry = {
                    "id": f"{filename}_{j}",
                    "title": title or filename,
                    "abstract": chunk,
                    "source_file": filename,
                    "chunk_index": j
                }
                all_data.append(data_entry)

            total_chunks += len(chunks)
            print(f"  → 提取成功: {len(chunks)} 个文本块")
        else:
            print(f"  → 内容过少或无内容，跳过")

    return all_data, total_chunks


def save_json_data(data, json_filepath):
    """保存JSON数据"""
    try:
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        file_size = os.path.getsize(json_filepath) / 1024  # KB
        print(f"✅ JSON保存成功: {json_filepath}")
        print(f"   数据条数: {len(data)}")
        print(f"   文件大小: {file_size:.1f} KB")

        return True
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False


def backup_chroma_data():
    """备份ChromaDB数据"""
    chroma_dir = "./chroma_data"
    if not os.path.exists(chroma_dir):
        print("ℹ️ ChromaDB数据目录不存在，无需备份")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"./backups/chroma_backup_{timestamp}"

    os.makedirs(os.path.dirname(backup_dir), exist_ok=True)

    try:
        shutil.copytree(chroma_dir, backup_dir)
        print(f"✅ ChromaDB数据备份到: {backup_dir}")
        return backup_dir
    except Exception as e:
        print(f"❌ 备份失败: {e}")
        return None


def clear_chroma_data():
    """清空ChromaDB数据"""
    chroma_dir = "./chroma_data"

    if os.path.exists(chroma_dir):
        try:
            shutil.rmtree(chroma_dir)
            os.makedirs(chroma_dir, exist_ok=True)
            print("✅ ChromaDB数据已清空")
            return True
        except Exception as e:
            print(f"❌ 清空失败: {e}")
            return False
    else:
        print("ℹ️ ChromaDB数据目录不存在，无需清空")
        return True


def main_menu():
    """主菜单"""
    print_header("医疗RAG系统 - 数据管理工具")

    while True:
        print("\n📋 主菜单:")
        print("  1. 📁 查看数据统计")
        print("  2. 🆕 生成示例HTML文件")
        print("  3. 🔄 重新处理所有HTML文件")
        print("  4. 💾 备份ChromaDB数据")
        print("  5. 🗑️  清空ChromaDB数据")
        print("  6. 🚀 启动应用")
        print("  7. 📤 导出数据统计")
        print("  0. 🔚 退出")

        choice = input("\n请选择操作 (0-7): ").strip()

        if choice == '1':
            view_data_stats()
        elif choice == '2':
            generate_html_menu()
        elif choice == '3':
            reprocess_data_menu()
        elif choice == '4':
            backup_chroma_data()
        elif choice == '5':
            clear_chroma_menu()
        elif choice == '6':
            launch_app()
        elif choice == '7':
            export_stats()
        elif choice == '0':
            print("\n👋 再见！")
            break
        else:
            print("❌ 无效选择，请重新输入")


def view_data_stats():
    """查看数据统计"""
    print_header("数据统计")

    html_files, json_file = check_data_directory()

    print(f"📁 HTML文件: {len(html_files)} 个")
    for i, file in enumerate(sorted(html_files)[:10], 1):
        size = os.path.getsize(f"./data/{file}") / 1024
        print(f"  {i:2d}. {file} ({size:.1f} KB)")

    if len(html_files) > 10:
        print(f"  ... 还有 {len(html_files) - 10} 个文件")

    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"\n📊 JSON数据: {len(data)} 条记录")

        # 统计各文件的块数
        from collections import Counter
        sources = Counter([item.get('source_file', '未知') for item in data])
        print("\n📈 各文件文本块分布:")
        for source, count in sources.most_common(10):
            print(f"  {source[:30]:30} : {count:3d} 块")
    else:
        print("\n❌ JSON文件不存在")

    # 检查ChromaDB
    if os.path.exists("./chroma_data"):
        print(f"\n🗄️  ChromaDB数据目录: 存在")
    else:
        print(f"\n🗄️  ChromaDB数据目录: 不存在")


def generate_html_menu():
    """生成HTML文件菜单"""
    print_header("生成示例HTML文件")

    created = generate_sample_html()
    if created:
        print(f"\n✅ 已创建 {len(created)} 个示例文件")
    else:
        print("ℹ️ 所有示例文件已存在")


def reprocess_data_menu():
    """重新处理数据菜单"""
    print_header("重新处理数据")

    # 备份确认
    print("⚠️  重新处理将生成新的JSON文件，可能会覆盖旧数据")
    confirm = input("是否继续? (y/N): ").lower()

    if confirm != 'y':
        print("❌ 已取消")
        return

    # 处理数据
    all_data, total_chunks = process_html_files()

    if not all_data:
        print("❌ 未生成任何数据")
        return

    # 保存JSON
    json_filepath = "./data/processed_data.json"
    if save_json_data(all_data, json_filepath):
        print(f"\n✅ 数据处理完成!")
        print(f"   总文本块数: {total_chunks}")
        print(f"   总数据条目: {len(all_data)}")

        # 建议清空ChromaDB
        print("\n💡 建议: 处理完新数据后，建议清空ChromaDB数据并重启应用")
    else:
        print("❌ 数据处理失败")


def clear_chroma_menu():
    """清空ChromaDB菜单"""
    print_header("清空ChromaDB数据")

    print("⚠️  清空后需要重新索引所有数据")
    confirm = input("确认清空? (y/N): ").lower()

    if confirm == 'y':
        if clear_chroma_data():
            print("\n✅ 请重启应用以重新索引数据")
        else:
            print("❌ 清空失败")
    else:
        print("❌ 已取消")


def launch_app():
    """启动应用"""
    print_header("启动应用")

    print("正在启动Streamlit应用...")
    print("请在新终端中运行: streamlit run app.py")
    print("或按 Ctrl+C 返回菜单")

    try:
        import subprocess
        subprocess.run(["streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n返回菜单...")


def export_stats():
    """导出统计信息"""
    print_header("导出统计")

    stats_file = f"./data/stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    html_files, json_file = check_data_directory()

    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("医疗RAG系统 - 数据统计\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"HTML文件数量: {len(html_files)}\n")
        f.write("HTML文件列表:\n")
        for file in sorted(html_files):
            size = os.path.getsize(f"./data/{file}") / 1024
            f.write(f"  - {file} ({size:.1f} KB)\n")

        f.write("\n")

        if os.path.exists(json_file):
            with open(json_file, 'r', encoding='utf-8') as jf:
                data = json.load(jf)

            f.write(f"JSON数据条目: {len(data)}\n")

            from collections import Counter
            sources = Counter([item.get('source_file', '未知') for item in data])

            f.write("\n各文件文本块分布:\n")
            for source, count in sources.most_common():
                f.write(f"  {source}: {count} 块\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("统计结束\n")

    print(f"✅ 统计信息已导出到: {stats_file}")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")
        import traceback

        traceback.print_exc()