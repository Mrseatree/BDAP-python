import fastapi
from fastapi import FastAPI
from consul_utils import register_service, deregister_service
from config import SERVICE_NAME
from typing import Optional, Union, Any, List
from pydantic import BaseModel
import atexit
import os
import json

app = FastAPI()


# 统一的参数模型（增强版本）
class UnifiedToolParams(BaseModel):
    # 文件内容字段
    file_content1: str = None
    file_content2: str = None
    
    # 使用字符串来存储动态参数，然后转换为字典
    params: Optional[str] = "{}"

    # 为了向后兼容，保留常用的直接参数
    column: Optional[str] = None
    old_name: Optional[str] = None
    new_name: Optional[str] = None
    condition: Optional[str] = None
    value: Optional[Union[str, int, float, bool]] = None
    target_type: Optional[str] = None
    group_by: Optional[str] = None
    target_column: Optional[str] = None
    agg_func: Optional[str] = None
    ascending: Optional[bool] = True
    constant_value: Optional[Union[str, int, float]] = None
    
    # 连接操作参数 - 设置默认值避免验证失败
    join_mode: Optional[str] = "inner"  # inner/left/right/outer
    on: Optional[str] = None
    left_on: Optional[str] = None
    right_on: Optional[str] = None
    
    # 添加更多可能的参数名称，让Dify更容易传参
    join_column: Optional[str] = None  # 别名for on
    connect_on: Optional[str] = None   # 别名for on
    merge_on: Optional[str] = None     # 别名for on

    def _parse_params(self) -> dict:
        """将字符串形式的params转换为字典"""
        try:
            if not self.params or self.params.strip() == "":
                return {}
            return json.loads(self.params)
        except json.JSONDecodeError as e:
            print(f"解析params参数失败: {e}, 使用空字典")
            return {}

    def get_param(self, key: str, default = None):
        """获取参数值，优先从直接参数获取，然后从params字典获取"""
        # 首先检查直接参数
        direct_value = getattr(self, key, None)
        if direct_value is not None:
            return direct_value

        # 然后从params字典中获取
        params_dict = self._parse_params()
        return params_dict.get(key, default)

    def get_join_column(self):
        """获取连接字段，尝试多个可能的参数名"""
        # 按优先级尝试不同的参数名
        candidates = [
            self.on,
            self.join_column, 
            self.connect_on,
            self.merge_on,
            self.get_param('on'),
            self.get_param('join_column'),
            self.get_param('connect_on'),
            self.get_param('merge_on'),
            self.get_param('id'),  # 常见的连接字段
            self.get_param('key'),
            self.get_param('join_key')
        ]
        
        for candidate in candidates:
            if candidate:
                return candidate
        return None

    def check_single_file(self):
        if not self.file_content1:
            raise ValueError("文件内容不得为空")

    def check_multi_files(self, max_files = 2):
        if not self.file_content1:
            raise ValueError("文件内容1不得为空")
        if not self.file_content2:
            raise ValueError("文件内容2不得为空")


# 添加服务启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """服务启动时注册到Consul"""
    service_id = start_tool_functions_service()
    if service_id:
        app.state.service_id = service_id
        print(f"tool_functions服务已注册到Consul，服务ID: {service_id}")


@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时从Consul注销"""
    if hasattr(app.state, 'service_id'):
        deregister_service(app.state.service_id)


def start_tool_functions_service():
    SERVICE_PORT = 8002
    tags = ['tools', 'data-processing', 'pandas']
    service_id = register_service(SERVICE_PORT, tags)
    return service_id


# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status":"healthy", "service":"tool-functions"}


@app.post("/tools/drop-empty-rows")
def drop_empty_rows(params: UnifiedToolParams) -> dict:
    import pandas as pd
    try:
        params.check_single_file()
        raw = params.file_content1

        data = json.loads(raw)
        # 如果还是 str，再转第二次
        if isinstance(data, str):
            data = json.loads(data)

        df = pd.DataFrame.from_records(data)
        cleaned_df = df.dropna()

        return {
            "status":"success",
            "message":"已删除所有空白行",
            "output_data":cleaned_df.to_dict(orient = "records")
        }
    except Exception as e:
        return {
            "status":"error",
            "message":f"去除空白行处理失败{str(e)}"
        }


@app.post("/tools/fill-missing-with-mean")
def fill_missing_with_mean(params: UnifiedToolParams) -> dict:
    import pandas as pd
    try:
        params.check_single_file()
        raw = params.file_content1

        data = json.loads(raw)
        if isinstance(data, str):
            data = json.loads(data)

        df = pd.DataFrame.from_records(data)
        df = df.fillna(df.mean(numeric_only = True))

        return {
            "status":"success",
            "message":"已使用平均值填补缺失值",
            "output_data":df.to_dict(orient = "records")
        }
    except Exception as e:
        return {
            "status":"error",
            "message":f"使用平均值填补缺失值处理失败:{str(e)}"
        }


@app.post("/tools/fill-missing-with-median")
def fill_missing_with_median(params: UnifiedToolParams) -> dict:
    import pandas as pd
    try:
        params.check_single_file()
        raw = params.file_content1

        data = json.loads(raw)
        if isinstance(data, str):
            data = json.loads(data)

        df = pd.DataFrame.from_records(data)
        df = df.fillna(df.median(numeric_only = True))

        return {
            "status":"success",
            "message":"已使用中位数填补缺失值",
            "output_data":df.to_dict(orient = "records")
        }
    except Exception as e:
        return {
            "status":"error",
            "message":f"使用中位数填补缺失值处理失败:{str(e)}"
        }


@app.post("/tools/fill-missing-with-constant")
def fill_missing_with_constant(params: UnifiedToolParams) -> dict:
    import pandas as pd
    try:
        params.check_single_file()
        raw = params.file_content1

        data = json.loads(raw)
        if isinstance(data, str):
            data = json.loads(data)

        df = pd.DataFrame.from_records(data)
        constant_value = params.get_param('constant_value') or params.get_param('value')
        if constant_value is None:
            raise ValueError("需要提供constant_value或value参数")

        df = df.fillna(constant_value)

        return {
            "status":"success",
            "message":"已使用常数填补缺失值",
            "output_data":df.to_dict(orient = "records")
        }
    except Exception as e:
        return {
            "status":"error",
            "message":f"使用常数填补缺失值处理失败:{str(e)}"
        }


@app.post("/tools/fill-missing-with-mode")
def fill_missing_with_mode(params: UnifiedToolParams) -> dict:
    import pandas as pd
    try:
        params.check_single_file()
        raw = params.file_content1

        data = json.loads(raw)
        if isinstance(data, str):
            data = json.loads(data)

        df = pd.DataFrame.from_records(data)
        df = df.fillna(df.mode().iloc[0])

        return {
            "status":"success",
            "message":"已使用众数填补缺失值",
            "output_data":df.to_dict(orient = "records")
        }
    except Exception as e:
        return {
            "status":"error",
            "message":f"使用众数填补缺失值处理失败:{str(e)}"
        }


@app.post("/tools/filter-by-column")
def filter_by_column(params: UnifiedToolParams) -> dict:
    import pandas as pd
    try:
        params.check_single_file()
        raw = params.file_content1

        data = json.loads(raw)
        if isinstance(data, str):
            data = json.loads(data)

        df = pd.DataFrame.from_records(data)
        column = params.get_param('column')
        condition = params.get_param('condition')
        value = params.get_param('value')

        if not column or not condition or value is None:
            raise ValueError("需要提供column、condition和value参数")

        if condition == '==':
            df = df[df[column] == value]
        elif condition == '!=':
            df = df[df[column] != value]
        elif condition == '>':
            df = df[df[column] > value]
        elif condition == '<':
            df = df[df[column] < value]
        elif condition == '>=':
            df = df[df[column] >= value]
        elif condition == '<=':
            df = df[df[column] <= value]
        else:
            raise ValueError("不支持的条件")

        return {
            "status":"success",
            "message":"已完成筛选",
            "output_data":df.to_dict(orient = "records")
        }
    except Exception as e:
        return {
            "status":"error",
            "message":f"筛选处理失败{str(e)}"
        }


@app.post("/tools/rename-column")
def rename_column(params: UnifiedToolParams) -> dict:
    import pandas as pd
    try:
        params.check_single_file()
        raw = params.file_content1

        data = json.loads(raw)
        if isinstance(data, str):
            data = json.loads(data)

        df = pd.DataFrame.from_records(data)
        old_name = params.get_param('old_name')
        new_name = params.get_param('new_name')

        if not old_name or not new_name:
            raise ValueError("需要提供old_name和new_name参数")
        if old_name not in df.columns:
            raise ValueError(f"列{old_name}不存在")

        df = df.rename(columns = {old_name:new_name})

        return {
            "status":"success",
            "message":"已完成重命名",
            "output_data":df.to_dict(orient = "records")
        }
    except Exception as e:
        return {
            "status":"error",
            "message":f"重命名处理失败{str(e)}"
        }


@app.post("/tools/convert-column-type")
def convert_column_type(params: UnifiedToolParams) -> dict:
    import pandas as pd
    try:
        params.check_single_file()
        raw = params.file_content1

        data = json.loads(raw)
        if isinstance(data, str):
            data = json.loads(data)

        df = pd.DataFrame.from_records(data)
        column = params.get_param('column')
        target_type = params.get_param('target_type')

        if not column or not target_type:
            raise ValueError("需要提供column和target_type参数")
        if column not in df.columns:
            raise ValueError(f"列{column}不存在")

        if target_type == "int":
            df[column] = df[column].astype(int)
        elif target_type == "float":
            df[column] = df[column].astype(float)
        elif target_type == "str":
            df[column] = df[column].astype(str)
        elif target_type == "bool":
            df[column] = df[column].astype(bool)
        else:
            raise ValueError("不支持的目标类型")

        return {
            "status":"success",
            "message":"已完成类型转换",
            "output_data":df.to_dict(orient = "records")
        }
    except Exception as e:
        return {
            "status":"error",
            "message":f"类型转换处理失败{str(e)}"
        }


@app.post("/tools/aggregate-column")
def aggregate_column(params: UnifiedToolParams) -> dict:
    import pandas as pd
    try:
        params.check_single_file()
        raw = params.file_content1

        data = json.loads(raw)
        if isinstance(data, str):
            data = json.loads(data)

        df = pd.DataFrame.from_records(data)
        group_by = params.get_param('group_by')
        target_column = params.get_param('target_column')  # 不聚合的列
        agg_func = params.get_param('agg_func')  # 聚合的列

        if not group_by:
            raise ValueError("需要提供 group_by 参数")

        # 支持逗号分隔字符串 → 列表
        if isinstance(group_by, str):
            group_by = [g.strip() for g in group_by.split(",")]

        # 构建聚合规则字典
        agg_dict = {}

        # target_column → 用 "first" 保留
        if target_column:
            if isinstance(target_column, str):
                target_column = [c.strip() for c in target_column.split(",")]
            for col in target_column:
                agg_dict[col] = "first"

        # agg_func → 聚合规则
        if agg_func:
            if isinstance(agg_func, str):
                if agg_func not in ['sum', 'mean', 'max', 'min', 'count']:
                    raise ValueError("不支持的聚合函数")
                # 如果 target_column 为空，就报错（因为需要知道作用在哪些列）
                if not target_column:
                    raise ValueError("使用字符串形式的 agg_func 时必须提供 target_column")
                for col in target_column:
                    agg_dict[col] = agg_func
            elif isinstance(agg_func, dict):
                agg_dict.update(agg_func)
            else:
                raise ValueError("agg_func 必须是字符串或字典")

        # 执行分组聚合
        if not agg_dict:
            # 没有任何聚合规则 → 只返回分组唯一组合
            result = df[group_by].drop_duplicates()
        else:
            result = df.groupby(group_by).agg(agg_dict).reset_index()

        return {
            "status":"success",
            "message":"已完成 aggregate 操作" + (" + 聚合" if agg_func else " (未聚合，仅输出分组和列)"),
            "output_data":result.to_dict(orient = "records")
        }
    except Exception as e:
        return {
            "status":"error",
            "message":f"aggregate 处理失败: {str(e)}"
        }


@app.post("/tools/sort-by-column")
def sort_by_column(params: UnifiedToolParams) -> dict:
    import pandas as pd
    try:
        params.check_single_file()
        raw = params.file_content1

        data = json.loads(raw)
        if isinstance(data, str):
            data = json.loads(data)

        df = pd.DataFrame.from_records(data)
        column = params.get_param('column')
        ascending = params.get_param('ascending', True)

        if not column:
            raise ValueError("需要提供column参数")
        if column not in df.columns:
            raise ValueError(f"列{column}不存在")

        df = df.sort_values(by = column, ascending = ascending)

        return {
            "status":"success",
            "message":"已完成排序",
            "output_data":df.to_dict(orient = "records")
        }
    except Exception as e:
        return {
            "status":"error",
            "message":f"排序处理失败{str(e)}"
        }


@app.post("/tools/join_tables")
def join_tables(params: UnifiedToolParams) -> dict:
    import pandas as pd
    try:
        # 检查必需的两个数据集
        params.check_multi_files()
        raw1 = params.file_content1
        raw2 = params.file_content2

        print(f"=== join_tables调试信息开始 ===")
        print(f"接收到的参数:")
        print(f"- file_content1 长度: {len(raw1) if raw1 else 0}")
        print(f"- file_content2 长度: {len(raw2) if raw2 else 0}")
        print(f"- join_mode: {params.join_mode}")
        print(f"- on: {params.on}")
        print(f"- join_column: {params.join_column}")
        print(f"- left_on: {params.left_on}")
        print(f"- right_on: {params.right_on}")
        
        # 解析数据
        data1 = json.loads(raw1)
        data2 = json.loads(raw2)

        # 处理可能的双重JSON编码
        if isinstance(data1, str):
            data1 = json.loads(data1)
        if isinstance(data2, str):
            data2 = json.loads(data2)

        df1 = pd.DataFrame.from_records(data1)
        df2 = pd.DataFrame.from_records(data2)
        
        print(f"- df1 shape: {df1.shape}, columns: {list(df1.columns)}")
        print(f"- df2 shape: {df2.shape}, columns: {list(df2.columns)}")
        
        # 获取连接模式，默认为inner
        how = params.join_mode or "inner"
        if how not in ["left", "right", "outer", "inner"]:
            raise ValueError("连接模式不合法，支持: inner, left, right, outer")

        # 获取连接字段 - 使用增强的方法
        on_field = params.get_join_column()
        left_on_field = params.left_on or params.get_param('left_on')
        right_on_field = params.right_on or params.get_param('right_on')

        print(f"- 解析后的连接字段:")
        print(f"  - on_field: {on_field}")
        print(f"  - left_on_field: {left_on_field}")
        print(f"  - right_on_field: {right_on_field}")

        # 执行连接操作
        result = None
        
        if on_field:
            # 使用相同的列名连接
            if on_field not in df1.columns:
                raise ValueError(f"左表中不存在列 '{on_field}', 可用列: {list(df1.columns)}")
            if on_field not in df2.columns:
                raise ValueError(f"右表中不存在列 '{on_field}', 可用列: {list(df2.columns)}")
            
            print(f"- 使用字段 '{on_field}' 进行 {how} 连接")
            result = pd.merge(df1, df2, how=how, on=on_field)
            
        elif left_on_field and right_on_field:
            # 使用不同的列名连接
            if left_on_field not in df1.columns:
                raise ValueError(f"左表中不存在列 '{left_on_field}', 可用列: {list(df1.columns)}")
            if right_on_field not in df2.columns:
                raise ValueError(f"右表中不存在列 '{right_on_field}', 可用列: {list(df2.columns)}")
            
            print(f"- 使用 left_on='{left_on_field}', right_on='{right_on_field}' 进行 {how} 连接")
            result = pd.merge(df1, df2, how=how, left_on=left_on_field, right_on=right_on_field)
            
        else:
            # 如果没有指定连接字段，尝试自动找到共同列
            common_cols = list(set(df1.columns) & set(df2.columns))
            print(f"- 共同列: {common_cols}")
            
            if not common_cols:
                raise ValueError(
                    f"未指定连接字段且两表没有共同列名。\n"
                    f"左表列: {list(df1.columns)}\n"
                    f"右表列: {list(df2.columns)}\n"
                    f"请在查询中明确指定连接字段，例如：'按照id字段连接'"
                )
            
            # 使用第一个共同列（优先选择id相关的列）
            auto_on = common_cols[0]
            for col in common_cols:
                if 'id' in col.lower():
                    auto_on = col
                    break
                    
            print(f"- 自动选择连接字段: '{auto_on}' (从共同列中选择)")
            result = pd.merge(df1, df2, how=how, on=auto_on)

        print(f"- 连接结果: {result.shape}, columns: {list(result.columns)}")
        print("=== join_tables调试信息结束 ===")

        return {
            "status": "success",
            "message": f"已完成 {how} join 操作，结果包含 {len(result)} 行 {len(result.columns)} 列数据",
            "output_data": result.to_dict(orient="records")
        }
        
    except Exception as e:
        print(f"join_tables 错误: {str(e)}")
        import traceback
        error_trace = traceback.format_exc()
        print(f"错误堆栈: {error_trace}")
        
        return {
            "status": "error",
            "message": f"join 处理失败: {str(e)}"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
