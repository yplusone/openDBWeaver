from .operator_gen import register_operator, build_operator, CodegenContext  # re-export

# 导入会触发各文件中的 @register_operator 装饰器
from . import aggregation
from . import input
from . import sort
from . import output
from . import filter
from . import define    
