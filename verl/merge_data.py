import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def merge_parquet_files_streaming(input_dir, output_file, batch_size=10000):
    """
    使用流式处理合并parquet文件，最节省内存
    """
    input_path = Path(input_dir)
    parquet_files = sorted([str(f) for f in input_path.glob("train_batch_*.parquet")])
    
    if not parquet_files:
        print("未找到匹配的parquet文件")
        return
    
    print(f"找到 {len(parquet_files)} 个文件")
    
    # 获取schema（从第一个文件）
    first_table = pq.read_table(parquet_files[0])
    schema = first_table.schema
    
    # 创建parquet写入器
    with pq.ParquetWriter(output_file, schema) as writer:
        for file_path in parquet_files:
            print(f"处理文件: {Path(file_path).name}")
            
            # 分批读取文件
            parquet_file = pq.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                writer.write_batch(batch)
    
    print("合并完成！")

# 使用示例
input_dir = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/guoxiaojun07/our_datasets/MMERealWorld_PositionQA"
merge_parquet_files_streaming(input_dir, "train.parquet", batch_size=5000)