import os
from dataclasses import dataclass, asdict

@dataclass
class Config:
    input_size: int = 512
    crop_size: int = 512
    backbone: str = "resnet18"
    layers: tuple = ("layer1", "layer2", "layer3")
    
    border_margin: int = 32
    sigma: float = 4.0
    min_defect_area: int = 50
    max_defect_coverage_ratio: float = 0.35
    
    data_dir: str = "/content/data/leather"
    checkpoint_path: str = "student_v2.pth"
    
    server_name: str = "0.0.0.0"
    server_port: int = 7860

    def to_dict(self):
        return asdict(self)
        
