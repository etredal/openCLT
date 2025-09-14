from dataclasses import dataclass, field


@dataclass
class TrainingMetric:
    total_loss:             list[float] = field(default_factory=list)
    reconstruction_loss:    list[float] = field(default_factory=list)
    sparsity_loss:          list[float] = field(default_factory=list)
    l0_metric:              list[float] = field(default_factory=list)
    learning_rate:          list[float] = field(default_factory=list) 