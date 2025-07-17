import yaml
import os
from pathlib import Path


def load_config(config_path = "faiss_config.yaml"):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding = 'utf-8') as f:
        config = yaml.safe_load(f)

    # 解析路径为绝对路径
    base_dir = Path(__file__).parent
    config['data']['input_path'] = str(base_dir / config['data']['input_path'])
    config['data']['output_db'] = str(base_dir / config['data']['output_db'])

    return config


def get_source_files(config):
    """获取所有待处理的PDF文件"""
    input_dir = Path(config['data']['input_path'])
    source_files = []
    for pattern in config['data']['file_patterns']:
        source_files.extend(input_dir.glob(pattern))
    return sorted(set(source_files), key = lambda x:x.name)
