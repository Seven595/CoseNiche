# 代码重构指南

本文档说明如何重构 tutorial 脚本，使其更加规范和易于使用。

## 🎯 重构目标

1. **消除硬编码路径** - 使用配置文件和命令行参数
2. **提取公共代码** - 移动到共享工具模块 (`utils/`)
3. **改进代码结构** - 使用清晰的函数分解
4. **增强错误处理** - 添加有意义的错误消息
5. **统一代码风格** - 遵循 Python 最佳实践

## 📂 新的目录结构

```
tutorials/
├── utils/                          # 共享工具模块
│   ├── __init__.py
│   ├── io.py                      # I/O 函数
│   ├── gene_filtering.py          # 基因过滤
│   ├── visualization.py           # 绘图工具
│   └── logger.py                  # 日志工具
│
├── deconvolution/
│   ├── run_deconvolution.py      # 重构后的主脚本
│   └── visualize_results.py      # 重构后的可视化脚本
│
├── attention_analysis/
│   └── ...（类似重构）
│
└── spatial_communication/
    └── ...（类似重构）
```

## 🔧 重构步骤

### 步骤 1: 使用配置文件

**之前（硬编码）**:
```python
h5ad_path = "/home/junning/projectnvme/ST/Data/PDAC/pdac.h5ad"
embeddings_path = "/home/junning/projectnvme/ST/.../embeddings.npy"
```

**之后（配置驱动）**:
```python
import argparse
from tutorials.utils import load_config

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, help='Path to config YAML')
args = parser.parse_args()

config = load_config(args.config)
h5ad_path = config['data']['h5ad_path']
embeddings_path = config['data']['embeddings_path']
```

### 步骤 2: 使用工具函数

**之前（重复代码）**:
```python
import scanpy as sc
adata = sc.read_h5ad(h5ad_path)
df_truth = pd.read_csv(truth_path)
adata.obs['ground_truth'] = df_truth['Region'].values
```

**之后（使用工具函数）**:
```python
from tutorials.utils import load_h5ad_with_truth

adata = load_h5ad_with_truth(
    h5ad_path=config['data']['h5ad_path'],
    truth_csv=config['data']['truth_csv'],
    truth_column='Region'
)
```

### 步骤 3: 提取基因过滤逻辑

**之前（混杂在主脚本中）**:
```python
def clean_symbol(s: str) -> str:
    # ... 长长的清洗逻辑 ...
    
def filter_symbols_and_attention(...):
    # ... 复杂的过滤逻辑 ...
```

**之后（使用工具模块）**:
```python
from tutorials.utils import clean_symbol, filter_genes_by_quality

symbols_clean, kept_indices = filter_genes_by_quality(
    symbols=raw_symbols,
    whitelist_prefixes=['HLA-', 'MIR'],
    blacklist_prefixes=['AC', 'AL', 'LINC']
)
```

### 步骤 4: 标准化可视化

**之前（重复的绘图代码）**:
```python
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    # ... 很多配置 ...
})
```

**之后（使用工具函数）**:
```python
from tutorials.utils import setup_plotting_style, save_figure

setup_plotting_style(style='nature', dpi=300)
fig, ax = plt.subplots()
# ... 绘图 ...
save_figure(fig, output_path, formats=['png', 'pdf'])
```

### 步骤 5: 添加日志

**之前（print 语句）**:
```python
print("Loading data...")
print(f"Loaded {n_obs} spots")
```

**之后（使用logger）**:
```python
from tutorials.utils import setup_logger

logger = setup_logger(
    name='deconvolution',
    level='INFO',
    log_file='./logs/deconvolution.log'
)

logger.info("Loading data...")
logger.info(f"Loaded {n_obs} spots")
```

## 📝 重构模板

### 主脚本模板

```python
#!/usr/bin/env python3
"""
Script description here.
"""

import argparse
from pathlib import Path
import sys

# Add tutorials to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tutorials.utils import (
    load_config,
    load_h5ad_with_truth,
    setup_logger,
    save_results
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Script description"
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use (overrides config)'
    )
    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    if args.device:
        config['training']['device'] = args.device
    
    # Setup logger
    logger = setup_logger(
        name='my_script',
        level=config.get('log_level', 'INFO'),
        log_file=Path(config['output']['output_dir']) / 'run.log'
    )
    
    logger.info("Starting analysis...")
    logger.info(f"Config: {args.config}")
    
    try:
        # Your analysis code here
        adata = load_h5ad_with_truth(
            h5ad_path=config['data']['h5ad_path'],
            truth_csv=config['data'].get('truth_csv')
        )
        
        # ... analysis ...
        
        # Save results
        save_results(results, output_path)
        
        logger.info("✓ Analysis completed successfully!")
    
    except Exception as e:
        logger.error(f"✗ Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
```

## 🚀 重构优先级

### 高优先级（立即进行）

1. **创建共享工具模块** ✅ 已完成
   - `utils/io.py`
   - `utils/gene_filtering.py`
   - `utils/visualization.py`
   - `utils/logger.py`

2. **重构反卷积脚本**
   - 移除硬编码路径
   - 使用配置文件
   - 添加命令行参数
   - 使用共享工具

3. **更新配置文件**
   - 添加所有必要的路径配置
   - 添加参数说明注释

### 中优先级（接下来进行）

4. **重构注意力分析脚本**
   - 统一基因过滤逻辑
   - 使用共享工具
   - 改进错误处理

5. **重构空间通讯脚本**
   - 简化数据导出流程
   - 统一可视化风格

### 低优先级（可选）

6. **创建测试**
   - 单元测试工具函数
   - 集成测试主脚本

7. **性能优化**
   - 并行处理
   - 内存优化

## 📋 重构检查清单

对于每个脚本，确保：

- [ ] 移除所有硬编码路径
- [ ] 添加命令行参数解析
- [ ] 使用配置文件
- [ ] 使用 `tutorials.utils` 中的工具函数
- [ ] 添加 logger 而不是 print
- [ ] 添加 try-except 错误处理
- [ ] 添加 docstring 文档
- [ ] 在函数中分解逻辑（每个函数做一件事）
- [ ] 更新相应的 README
- [ ] 测试脚本是否正常运行

## 🔄 迁移示例

### 示例 1: 反卷积主脚本

查看 `deconvolution/run_deconvolution.py`（重构后的版本）

### 示例 2: 注意力导出脚本

查看 `attention_analysis/export_attention.py`（重构后的版本）

## 💡 最佳实践

1. **配置优先**: 所有路径和参数放在配置文件中
2. **命令行覆盖**: 允许命令行参数覆盖配置
3. **错误处理**: 使用 try-except 并提供有用的错误消息
4. **日志记录**: 使用 logger 而不是 print
5. **函数分解**: 每个函数只做一件事
6. **类型提示**: 添加类型提示提高可读性
7. **文档字符串**: 为所有函数添加 docstring

## 📞 获取帮助

如有问题，请参考：
- 重构后的示例脚本
- `utils/` 模块的文档字符串
- 项目主 README

---

**更新日期**: 2024-01-28
