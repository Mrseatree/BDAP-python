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
        self.node_map = {}
        self.components_config = loadComponentConfig("./component_whitelist.json")

    def sanitize(self, workflow_data: dict) -> Tuple[dict, List[str], List[str]]:
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
        #     self.warnings.append(f"节点数量超过限制({self.max_nodes}),已截断")
        #     workflow_data['nodes'] = nodes[:self.max_nodes]

        # 3. 验证并修正每个节点
        valid_nodes = []
        seen_marks = set()

        for node in nodes:
            node_mark = node.get('mark', '')

            # 检查必需字段
            if not self._validate_node_structure(node):
                continue

            # 检查节点mark唯一性
            if node_mark in seen_marks:
                self.warnings.append(f"节点标识冲突: {node_mark}")
                continue

            seen_marks.add(node_mark)

            # 组件白名单验证
            if node.get('id') not in self.whitelist:
                self.errors.append(f"无效的组件名: '{node.get('id')}'")
                continue

            # 验证锚点
            self._init_anchors(node)

            valid_nodes.append(node)
            self.node_map[node_mark] = node

        workflow_data['nodes'] = valid_nodes

        # 4. 验证连接关系
        for node in valid_nodes:
            self._validate_connections(node)

        # 5. 检测循环
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
        """验证节点基本结构 - 根据新格式只需要id和mark"""
        required_fields = {'id', 'mark'}
        missing_fields = required_fields - node.keys()
        
        if missing_fields:
            node_name = node.get('id', 'unknown')
            self.warnings.append(f"节点 '{node_name}' 缺少必需字段: {missing_fields}")
            return False
        return True

    def _init_anchors(self, node: dict):
        """初始化锚点结构"""
        # 输入锚点
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

        # 输出锚点
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

    def _validate_connections(self, node: dict):
        """验证连接关系"""
        node_mark = node.get('mark', '')

        # 验证输入连接
        for anchor in node['inputAnchors']:
            if 'sourceAnchor' in anchor and anchor['sourceAnchor']:
                source_anchor = anchor['sourceAnchor']
                source_mark = str(source_anchor.get('nodeMark', ''))
                
                if source_mark and source_mark not in self.node_map:
                    self.warnings.append(f"节点 {node_mark} 引用了不存在的源节点: {source_mark}")
                    # 清除无效连接
                    anchor['sourceAnchor'] = None
                    anchor['numOfConnectedEdges'] = 0

        # 验证输出连接
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
