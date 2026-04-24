"""
光学暗角与像差合成管线
=====================================
Vignette and Aberration Synthesis Pipeline

该模块提供高精度光学暗角与像差合成功能，用于生成包含复杂物理逻辑
与随机性扰动的暗角及像差合成数据集。

主要功能:
- 基于 cos^4 定律的暗角生成
- 高频像散与径向色散模拟
- 多角度裁剪与蒙版应用
- 动态噪点注入
- Ground Truth 提取

使用示例:
    from vignette_aberration_synthesizer import run_synthesis
    
    # 生成 400 组数据
    results = run_synthesis(num_groups=400, output_dir="output")
"""

from .config import (
    SynthesisConfig,
    VignetteConfig,
    AberrationConfig,
    CropConfig,
    NoiseConfig,
    LightSpotConfig,
)

from .synthesizer import (
    VignetteAberrationSynthesizer,
    run_synthesis,
    SynthesisResult,
)

from .vignette_generator import (
    generate_cos4_vignette,
    create_composite_mask,
    rotate_mask,
)

from .base_processor import (
    generate_light_spots,
    lock_base_image,
    crop_at_angle,
    create_base_image,
)

from .noise_injector import (
    apply_noise,
    apply_noise_fixed_ratio,
    generate_noise_intensity_ratio,
)

__version__ = "1.0.0"
__all__ = [
    # 配置
    "SynthesisConfig",
    "VignetteConfig", 
    "AberrationConfig",
    "CropConfig",
    "NoiseConfig",
    "LightSpotConfig",
    # 核心合成器
    "VignetteAberrationSynthesizer",
    "run_synthesis",
    "SynthesisResult",
    # 工具函数
    "generate_cos4_vignette",
    "create_composite_mask",
    "rotate_mask",
    "generate_light_spots",
    "lock_base_image",
    "crop_at_angle",
    "create_base_image",
    "apply_noise",
    "apply_noise_fixed_ratio",
    "generate_noise_intensity_ratio",
]