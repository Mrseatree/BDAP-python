import os
import hashlib
import datetime
import json
import numpy as np
import faiss
import yaml
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer


class RecursiveCharacterTextSplitter:
    """递归字符文本分割器，支持多级分隔符"""

    def __init__(self, separators: List[str], chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split_text(self, text: str) -> List[str]:
        """递归分割文本"""
        # 实现递归分割逻辑（这里简化为按长度分割）
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
        return chunks


class VectorStoreBuilder:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_model = self._init_embedding_model()
        self.text_splitter = self._init_text_splitter()

    def _init_embedding_model(self) -> SentenceTransformer:
        """初始化本地 BGE-m3 嵌入模型"""
        embedding_config = self.config['embedding']
        model_path = embedding_config.get('model_path', embedding_config['model_name'])
        device = embedding_config.get('device', 'cpu')

        return SentenceTransformer(
            model_name_or_path = model_path,
            device = device
        )

    def _init_text_splitter(self):
        """初始化文本分割器"""
        processing = self.config['processing']
        return RecursiveCharacterTextSplitter(
            separators = processing['separators'],
            chunk_size = processing['chunk_size'],
            chunk_overlap = processing['chunk_overlap']
        )

    def _extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        """提取文件元数据"""
        stat = file_path.stat()
        metadata_config = self.config['metadata']

        metadata = {
            "source":file_path.name,  # 默认使用文件名
            "file_id":hashlib.md5(str(file_path).encode()).hexdigest()[:8],
            "file_size":stat.st_size,
            "created_time":datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_time":datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "file_type":file_path.suffix.lower()[1:],
            "title":metadata_config.get('default_title', '')
        }

        # 如果配置了source_field，则覆盖source
        if 'source_field' in metadata_config:
            if metadata_config['source_field'] == 'file_name':
                metadata['source'] = file_path.name
            # 其他字段可以扩展

        return metadata

    def _read_file(self, file_path: Path) -> str:
        """读取文件内容"""
        # PDF文件
        if file_path.suffix.lower() == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    return '\n'.join(page.extract_text() for page in reader.pages)
            except ImportError:
                print("PyPDF2未安装，跳过PDF文件")
                return ""

        # Word文档
        if file_path.suffix.lower() in ['.docx']:
            try:
                from docx import Document
                doc = Document(file_path)
                return '\n'.join(para.text for para in doc.paragraphs)
            except ImportError:
                print("python-docx未安装，跳过Word文档")
                return ""

        # 其他文件类型跳过
        print(f"不支持的文件类型: {file_path.suffix}")
        return ""

    def process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """处理单个文件"""
        print(f"处理文件: {file_path.name}")
        try:
            content = self._read_file(file_path)
            if not content.strip():
                print(f"文件内容为空: {file_path.name}")
                return []

            base_meta = self._extract_metadata(file_path)
            chunks = self.text_splitter.split_text(content)

            results = []
            for i, chunk in enumerate(chunks):
                results.append({
                    "id":f"{base_meta['file_id']}-{i}",
                    "content":chunk,
                    "metadata":{**base_meta, "chunk_index":i}
                })

            print(f"生成 {len(results)} 个文本块")
            return results
        except Exception as e:
            print(f"处理文件失败: {file_path.name}, 错误: {str(e)}")
            return []

    def build_vector_store(self):
        """构建向量存储"""
        data_config = self.config['data']

        # 获取所有文件
        source_dir = Path(data_config['input_path'])
        file_patterns = data_config['file_pattern']

        file_paths = []
        for pattern in file_patterns:
            file_paths.extend(list(source_dir.glob(pattern)))

        if not file_paths:
            print(f"在 {source_dir} 中未找到匹配文件")
            return None

        # 处理所有文件
        all_chunks = []
        for file_path in file_paths:
            if file_path.is_file():
                chunks = self.process_file(file_path)
                if chunks:
                    print(f"已处理: {file_path.name} -> {len(chunks)} 个文本块")
                    all_chunks.extend(chunks)

        if not all_chunks:
            print("未提取到有效文本块")
            return None

        # 提取文本和元数据
        texts = [chunk["content"] for chunk in all_chunks]
        metadata = [chunk["metadata"] for chunk in all_chunks]

        # 生成嵌入向量
        print(f"为 {len(texts)} 个文本块生成嵌入...")
        embedding_config = self.config['embedding']
        encode_kwargs = embedding_config.get('encode_kwargs', {})

        embeddings = self.embedding_model.encode(
            texts,
            batch_size = 32,
            show_progress_bar = True,
            normalize_embeddings = encode_kwargs.get('normalize_embeddings', True)
        )
        embeddings = embeddings.astype(np.float32)

        # 创建FAISS索引
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        # 保存向量索引和元数据
        output_db = Path(data_config['output_db'])
        output_db.parent.mkdir(parents = True, exist_ok = True)

        # 保存FAISS索引
        faiss.write_index(index, str(output_db))

        # 保存元数据
        metadata_file = output_db.with_suffix('.json')
        with open(metadata_file, 'w', encoding = 'utf-8') as f:
            json.dump(metadata, f, ensure_ascii = False, indent = 2)

        print(f"\n向量存储构建完成!")
        print(f"FAISS索引文件: {output_db}")
        print(f"元数据文件: {metadata_file}")
        print(f"总文档块数: {len(metadata)}")
        print(f"嵌入维度: {dimension}")

        return {
            "index_file":str(output_db),
            "metadata_file":str(metadata_file),
            "total_chunks":len(metadata),
            "embedding_dim":dimension
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """从YAML文件加载配置"""
    if not Path(config_path).exists():
        print(f"配置文件 {config_path} 不存在!")
        return None

    with open(config_path, 'r', encoding = 'utf-8') as f:
        config = yaml.safe_load(f)

    return config


if __name__ == "__main__":
    # 加载配置文件
    config_path = "faiss_config.yaml"
    config = load_config(config_path)

    if not config:
        print("无法加载配置，程序退出")
        exit(1)

    print("配置加载成功:")
    print(f"输入路径: {config['data']['input_path']}")
    print(f"文件模式: {config['data']['file_pattern']}")
    print(f"输出路径: {config['data']['output_db']}")
    print(f"模型路径: {config['embedding'].get('model_path', config['embedding']['model_name'])}")
    print(f"设备: {config['embedding']['device']}")

    builder = VectorStoreBuilder(config)
    result = builder.build_vector_store()

    if result:
        print("\n向量库构建成功!")
        print(f"FAISS索引已保存到: {result['index_file']}")
        print(f"元数据已保存到: {result['metadata_file']}")