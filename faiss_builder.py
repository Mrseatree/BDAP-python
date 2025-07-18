import datetime
import hashlib
import os

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredFileLoader
)
from pathlib import Path
from typing import List, Dict, Any
from config_loader import load_config,get_source_files


class VectorStoreBuilder:
    def __init__(self,config):
        self.config = config
        self.embedding_model =self._init_embedding_model()
        self.text_splitter = self._init_text_splitter()

    def _init_embedding_model(self):
        model_type = self.config['embedding']['model_type']
        if model_type == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=self.config['embedding']['model_name'],
                model_kwargs={'device': 'cpu'}  # 改为'cuda'使用GPU
            )
        elif model_type == "openai":
            return OpenAIEmbeddings(
                openai_api_key=self.config['embedding']['openai_api_key'],
                model="text-embedding-3-small"
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

    def _init_text_splitter(self):
        return RecursiveCharacterTextSplitter(
            chunk_size = self.config['processing']['chunk_size'],
            chunk_overlap = self.config['processing']['chunk_overlap'],
            separators = self.config['processing']['separators']
        )

    def _extract_metadata(self, file_path):
        """从文件路径提取元数据"""
        file_name = file_path.name
        file_size = file_path.stat().st_size
        created_time = datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()

        # 生成唯一文件ID
        file_id = hashlib.md5(file_name.encode()).hexdigest()[:8]

        return {
            "source":file_name,
            "file_id":file_id,
            "file_size":file_size,
            "created_time":created_time,
            "title":self.config['metadata']['default_title'],
            "file_type":file_path.suffix.lower()[1:]  # 如 "pdf", "docx"
        }

    # 知识库文本
    def _process_pdf(self, pdf_path):
        """处理单个PDF文件"""
        print(f"处理文档: {os.path.basename(pdf_path)}")

        try:
            # 加载PDF
            loader = PyPDFLoader(str(pdf_path))
            raw_docs = loader.load()

            # 基础元数据
            base_metadata = self._extract_metadata(pdf_path)

            # 处理所有页面
            all_docs = []
            for i, page in enumerate(raw_docs):
                # 分割文本
                chunks = self.text_splitter.split_text(page.page_content)

                # 为每个块创建文档
                for j, chunk in enumerate(chunks):
                    metadata = {
                        **base_metadata,
                        "page":page.metadata['page'] + 1,  # 页码从1开始
                        "chunk_id":f"{i + 1}-{j + 1}",
                        "total_chunks":len(chunks)
                    }
                    all_docs.append({
                        "content":chunk,
                        "metadata":metadata
                    })

            print(f"生成 {len(all_docs)} 个文本块")
            return all_docs
        except Exception as e:
            print(f"处理PDF失败: {pdf_path.name}, 错误: {str(e)}")
            return []

    def _process_docx(self, docx_path: Path) -> List[Dict]:
        """处理Word文档 (.docx)"""
        print(f"处理Word文档: {docx_path.name}")

        try:
            # 使用UnstructuredWordDocumentLoader加载文档
            loader = UnstructuredWordDocumentLoader(
                str(docx_path),
                mode = "elements",  # 返回结构化元素
                strategy = "fast"  # 快速模式
            )
            elements = loader.load()

            # 基础元数据
            base_metadata = self._extract_metadata(docx_path)

            # 处理文档元素
            all_docs = []
            current_hierarchy = []  # 用于跟踪标题层级

            for element in elements:
                element_type = element.metadata.get("category", "unstructured")
                text_content = element.page_content.strip()

                if not text_content:
                    continue

                # 处理标题元素
                if element_type == "Title":
                    base_metadata["title"] = text_content

                # 处理标题层级
                elif element_type.startswith("Heading"):
                    try:
                        level = int(element_type.replace("Heading", ""))
                    except ValueError:
                        level = 1

                    # 更新标题层级
                    if level <= len(current_hierarchy):
                        current_hierarchy = current_hierarchy[:level - 1]
                    current_hierarchy.append(text_content)

                # 处理正文内容
                elif element_type in ["NarrativeText", "UncategorizedText"]:
                    # 创建元数据副本并添加标题信息
                    metadata = base_metadata.copy()

                    # 添加标题层级信息
                    max_levels = self.config['metadata'].get('preserve_heading_levels', 3)
                    for i, heading in enumerate(current_hierarchy[:max_levels]):
                        metadata[f"heading_{i + 1}"] = heading

                    # 分割文本
                    chunks = self.text_splitter.split_text(text_content)

                    # 为每个块创建文档
                    for j, chunk in enumerate(chunks):
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_id"] = f"{len(all_docs) + 1}-{j + 1}"
                        all_docs.append({
                            "content":chunk,
                            "metadata":chunk_metadata
                        })

            print(f"生成 {len(all_docs)} 个文本块")
            return all_docs

        except Exception as e:
            print(f"处理Word文档失败: {docx_path.name}, 错误: {str(e)}")
            return []

    def process_file(self,file_path:Path)->List[Dict]:
        extension = file_path.suffix.lower()

        if extension == '.pdf':
            return self._process_pdf(file_path)
        elif extension == '.docx':
            return self._process_docx(file_path)
        else:
            print(f"跳过不支持的文件类型: {file_path}")
            return []

    # 主处理流程
    def build_vector_store(self):
        """构建向量存储"""
        source_files = get_source_files(self.config)
        if not source_files:
            print("未找到匹配的PDF文件")
            return None

        # 处理所有文档
        all_documents = []
        for file_path in source_files:
            file_docs = self.process_file(file_path)
            all_documents.extend(file_docs)
            print(f"已处理: {file_path.name} -> {len(file_docs)} 个文本块")

        # 转换为LangChain文档格式
        langchain_docs = [
            Document(page_content = doc["content"], metadata = doc["metadata"])
            for doc in all_documents
        ]

        # 创建向量存储
        print(f"正在创建向量索引 ({len(langchain_docs)} 个文档块)...")
        vector_store = FAISS.from_documents(langchain_docs, self.embedding_model)

        # 保存结果
        output_path = self.config['data']['output_db']
        os.makedirs(os.path.dirname(output_path), exist_ok = True)
        vector_store.save_local(output_path)
        print(f"向量库已保存至: {output_path}")
        return vector_store


# 使用示例
if __name__ == "__main__":
    # 配置路径
    config = load_config()
    # 执行处理流程
    builder = VectorStoreBuilder(config)
    vector_db = builder.build_vector_store()

