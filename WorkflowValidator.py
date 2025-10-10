import json
import re
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Union, Any
from load import loadComponentConfig, loadWhiteList
import asyncio


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
            local_warnings, local_errors = [], []
            node_mark = node.get("mark", "")
            if not self._validate_node_structure(node):
                return None, local_warnings, local_errors

            if node_mark in seen_marks:
                local_warnings.append(f"节点标识冲突: {node_mark}")
                return None, local_warnings, local_errors

            if node.get("id") not in self.whitelist:
                local_errors.append(f"无效的组件名: '{node.get('id')}'")
                return None, local_warnings, local_errors

            # 不再进行属性处理，因为新结构移除了属性字段
            self._init_anchors(node)
            return node, local_warnings, local_errors

        results = await asyncio.gather(*(process_node(n) for n in nodes))

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
        """验证节点基本结构 - 更新为新的字段要求"""
        required_fields = {'id', 'mark'}  # 移除了 position 字段
        missing_fields = required_fields - node.keys()
        
        if missing_fields:
            node_name = node.get('id', 'unknown')
            self.warnings.append(f"节点 '{node_name}' 缺少必需字段: {missing_fields}")
            return False
        
        # 确保必要的锚点字段存在
        if 'inputAnchors' not in node:
            node['inputAnchors'] = []
        if 'outputAnchors' not in node:
            node['outputAnchors'] = []
            
        return True

    def _init_anchors(self, node: dict):
        """初始化锚点结构 - 适配新格式"""
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
