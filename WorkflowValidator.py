import json
import re
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Union, Any
from load import loadComponentConfig, loadWhiteList


class SimplifiedWorkflowValidator:
    def __init__(self, max_nodes: int = 10):
        self.whitelist = loadWhiteList("./component_whitelist.json")
        self.max_nodes = max_nodes
        self.warnings = []
        self.errors = []
        self.node_map = {}  # 使用 mark 作为键
        self.components_config = loadComponentConfig("./component_whitelist.json")

    async def sanitize(self, workflow_data: dict) -> Tuple[dict, List[str], List[str]]:
        # 重置状态
        self.warnings = []
        self.errors = []
        self.node_map = {}

        # 1. 验证基本结构
        if not self._validate_basic_structure(workflow_data):
            return None, self.warnings, self.errors

        # 2. 处理节点
        nodes = workflow_data.get('nodes', [])
        # TODO:节点个数暂时不做限制
        # if len(nodes) > self.max_nodes:
        #     self.warnings.append(f"节点数量超过限制({self.max_nodes})，已截断")
        #     workflow_data['nodes'] = nodes[:self.max_nodes]

        # 3. 验证并修正每个节点
        valid_nodes = []
        seen_marks = set()

        # 异步处理每个节点
        async def process_node(node):
            local_warnings,local_errors=[],[]
            node_mark=node.get("mark","")
            if not self._validate_node_structure(node):
                return None, local_warnings, local_errors

            if node_mark in seen_marks:
                local_warnings.append(f"节点标识冲突: {node_mark}")
                return None, local_warnings, local_errors

            if node.get("id") not in self.whitelist:
                local_errors.append(f"无效的组件名: '{node.get('id')}'")
                return None, local_warnings, local_errors

            await self._sanitize_attributes(node)  # 异步属性处理
            self._init_anchors(node)
            return node, local_warnings, local_errors

        results=await asyncio.gather(*(process_node(n)for n in nodes))

        for node, local_warnings, local_errors in results:
            self.warnings.extend(local_warnings)
            self.errors.extend(local_errors)
            if node:
                mark = node["mark"]
                if mark not in seen_marks:
                    seen_marks.add(mark)
                    valid_nodes.append(node)
                    self.node_map[mark] = node

        workflow_data["nodes"] = valid_nodes

        # 异步检查节点之间的连接
        await asyncio.gather(*(self._validate_connections(node) for node in valid_nodes))

        # 同步检测循环依赖
        if self._detect_cycles():
            self.errors.append("工作流中存在循环依赖")
            return None, self.warnings, self.errors

        return workflow_data, self.warnings, self.errors


    def _validate_basic_structure(self, data: dict) -> bool:
        """验证根结构完整性"""
        required_keys = {"requestId", "conversation_id", "nodes"}
        if not required_keys.issubset(data.keys()):
            missing_keys = required_keys - data.keys()
            self.errors.append(f"缺少必需的根字段: {missing_keys}")
            return False

        if not isinstance(data['nodes'], list):
            self.errors.append("nodes字段必须是列表类型")
            return False

        return True

    def _validate_node_structure(self, node: dict) -> bool:
        """验证节点基本结构"""
        required_fields = {'id', 'mark', 'position'}
        missing_fields = required_fields - node.keys()
        
        if missing_fields:
            node_name = node.get('name', node.get('id', 'unknown'))
            self.warnings.append(f"节点 '{node_name}' 缺少必需字段: {missing_fields}")
            return False
        return True

    def _init_anchors(self, node: dict):
        """初始化锚点结构"""
        # 输入锚点 - 新格式
        node.setdefault('inputAnchors', [])
        for anchor in node['inputAnchors']:
            anchor.setdefault('seq', 0)
            anchor.setdefault('numOfConnectedEdges', 0)
            
            # 确保sourceAnchor的格式正确
            if 'sourceAnchor' in anchor and anchor['sourceAnchor']:
                source_anchor = anchor['sourceAnchor']
                source_anchor.setdefault('nodeName', '')
                source_anchor.setdefault('nodeMark', 0)
                source_anchor.setdefault('seq', 0)
                
                # 确保nodeMark是整数类型
                try:
                    source_anchor['nodeMark'] = int(source_anchor['nodeMark'])
                except (ValueError, TypeError):
                    source_anchor['nodeMark'] = 0

        # 输出锚点 - 新格式
        node.setdefault('outputAnchors', [])
        for anchor in node['outputAnchors']:
            anchor.setdefault('seq', 0)
            anchor.setdefault('numOfConnectedEdges', 0)
            anchor.setdefault('targetAnchors', [])
            
            for target_anchor in anchor['targetAnchors']:
                target_anchor.setdefault('nodeName', '')
                target_anchor.setdefault('nodeMark', 0)
                target_anchor.setdefault('seq', 0)
                
                # 确保nodeMark是整数类型
                try:
                    target_anchor['nodeMark'] = int(target_anchor['nodeMark'])
                except (ValueError, TypeError):
                    target_anchor['nodeMark'] = 0

    async def _validate_connections(self, node: dict):
        """验证连接关系"""
        node_mark = node.get('mark', '')

        # 验证输入连接 - 新格式
        for anchor in node['inputAnchors']:
            if 'sourceAnchor' in anchor and anchor['sourceAnchor']:
                source_anchor = anchor['sourceAnchor']
                source_mark = str(source_anchor.get('nodeMark', ''))
                
                if source_mark and source_mark in self.node_map:
                    anchor['numOfConnectedEdges'] = 1
                elif source_mark:
                    self.warnings.append(f"节点 {node_mark} 引用了不存在的源节点: {source_mark}")
                    # 清除无效连接
                    anchor['sourceAnchor'] = None
                    anchor['numOfConnectedEdges'] = 0
                else:
                    anchor['numOfConnectedEdges'] = 0
            else:
                anchor['numOfConnectedEdges'] = 0

        # 验证输出连接 - 新格式
        for anchor in node['outputAnchors']:
            valid_targets = []
            
            for target_anchor in anchor.get('targetAnchors', []):
                target_mark = str(target_anchor.get('nodeMark', ''))
                
                if target_mark and target_mark in self.node_map:
                    valid_targets.append(target_anchor)
                elif target_mark:
                    self.warnings.append(f"节点 {node_mark} 引用了不存在的目标节点: {target_mark}")

            anchor['targetAnchors'] = valid_targets
            anchor['numOfConnectedEdges'] = len(valid_targets)

    def _detect_cycles(self) -> bool:
        """检测循环依赖"""
        # 构建连接图 - 使用mark作为节点标识
        graph = defaultdict(list)
        
        for node_mark, node in self.node_map.items():
            for anchor in node.get('inputAnchors', []):
                if 'sourceAnchor' in anchor and anchor['sourceAnchor']:
                    source_mark = str(anchor['sourceAnchor'].get('nodeMark', ''))
                    if source_mark in self.node_map:
                        graph[source_mark].append(node_mark)

        # 使用DFS检测循环
        visited = set()
        rec_stack = set()

        def dfs(mark):
            visited.add(mark)
            rec_stack.add(mark)

            for neighbor in graph.get(mark, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(mark)
            return False

        for node_mark in self.node_map.keys():
            if node_mark not in visited:
                if dfs(node_mark):
                    return True

        return False

    async def _sanitize_attributes(self, node: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        根据节点信息校验属性是否符合规范
        Args:
            node: 节点字典，包含id和attributes等信息
        Returns:
            包含错误信息的字典，键为错误类型，值为错误消息列表
        """
        # 获取组件名和属性
        component_name = node.get("id")

        simple_attrs = {}
        complicated_attrs = {}
        
        # 处理simpleAttributes列表
        for attr in node.get("simpleAttributes", []):
            if isinstance(attr, dict) and "name" in attr:
                simple_attrs[attr["name"]] = attr.get("value", "")
        
        # 处理complicatedAttributes列表
        for attr in node.get("complicatedAttributes", []):
            if isinstance(attr, dict) and "name" in attr:
                complicated_attrs[attr["name"]] = attr.get("value", "")
        
        provided_attrs = {**simple_attrs, **complicated_attrs}

        # 查找组件配置
        component_config = self.components_config.get(component_name)
        if not component_config:
            node_name = node.get("name", "未知节点")
            self.warnings.append(f"节点 '{node_name}': 未找到id为 '{component_name}' 的组件配置")
            return {"component_not_found": [f"节点 '{node_name}': 未找到id为 '{component_name}' 的组件配置"]}

        errors = {
            "missing_required": [],  # 缺失必填参数
            "unknown_attributes": [],  # 未知参数
            "type_mismatch": [],  # 类型不匹配
            "invalid_option": []  # 选项值无效
        }

        # 获取所有已知属性名
        known_simple_attrs = {attr["name"]: attr for attr in component_config.get("simpleAttributes", [])}
        known_complex_attrs = {attr["name"]: attr for attr in component_config.get("complicatedAttributes", [])}
        all_known_attrs = {**known_simple_attrs, **known_complex_attrs}

        # 检查是否有未知属性
        for attr_name in provided_attrs.keys():
            if attr_name not in all_known_attrs:
                errors["unknown_attributes"].append(f"未知参数: '{attr_name}'")

        # 检查必填属性（简单属性）是否都存在
        for attr_name, attr_config in known_simple_attrs.items():
            if attr_name not in provided_attrs:
                chinese_name = attr_config.get("chineseName", attr_name)
                errors["missing_required"].append(f"缺失必填参数: '{chinese_name}'({attr_name})")

        # 检查提供的属性值类型和选项
        for attr_name, attr_value in provided_attrs.items():
            if attr_name not in all_known_attrs:
                continue  # 已经在前面处理过未知属性

            attr_config = all_known_attrs[attr_name]
            expected_type = attr_config.get("valueType")
            allowed_options = attr_config.get("options")

            # 类型检查
            if expected_type and not self._check_type(attr_value, expected_type):
                chinese_name = attr_config.get("chineseName", attr_name)
                errors["type_mismatch"].append(
                    f"参数 '{chinese_name}' 类型错误: 期望 {expected_type}, 实际 {type(attr_value).__name__}"
                )

            # # 选项检查（仅适用于有预定义选项的参数）
            # if allowed_options and attr_value not in allowed_options:
            #     chinese_name = attr_config.get("chineseName", attr_name)
            #     errors["invalid_option"].append(
            #         f"参数 '{chinese_name}' 值 '{attr_value}' 无效，可选值: {allowed_options}"
            #     )

        # 将错误信息添加到警告列表
        for error_type, error_messages in errors.items():
            if error_messages:
                for error_msg in error_messages:
                    self.warnings.append(f"节点 '{node.get('name', node.get('id', 'unknown'))}': {error_msg}")

        # 移除空错误列表
        return {k: v for k, v in errors.items() if v}

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """
        检查值是否符合预期的类型
        Args:
            value: 要检查的值
            expected_type: 期望的类型字符串
        Returns:
            类型是否匹配
        """
        type_mapping = {
            "String": str,
            "Int": int,
            "Double": float,
            "Boolean": bool,
            "Long": int
        }

        expected_python_type = type_mapping.get(expected_type)
        if not expected_python_type:
            return True  # 未知类型，跳过检查

        # 特殊处理：Int和Long类型也接受字符串形式的数字
        if expected_type in ["Int", "Long"] and isinstance(value, str):
            try:
                int(value)
                return True
            except ValueError:
                return False

        # 特殊处理：Double类型也接受字符串形式的数字
        if expected_type == "Double" and isinstance(value, str):
            try:
                float(value)
                return True
            except ValueError:
                return False

        # 特殊处理：Boolean类型也接受字符串形式的布尔值
        if expected_type == "Boolean" and isinstance(value, str):
            return value.lower() in ["true", "false", "1", "0"]

        # 常规类型检查
        return isinstance(value, expected_python_type)
