"""Pre-mock heavy modules that hang on M1 Mac during import.

torch_geometric JIT-compiles C++ extensions on first import, which can take
minutes or hang entirely. Since our encoder unit tests only need torch_frame
(not torch_geometric, relbench, or sentence_transformers), we insert mocks
into sys.modules before any test file triggers `import encoders`.
"""

import sys
from unittest.mock import MagicMock

_HEAVY_MODULES = [
    "torch_geometric",
    "torch_geometric.data",
    "torch_geometric.nn",
    "torch_geometric.transforms",
    "sentence_transformers",
    "relbench",
    "relbench.base",
    "relbench.modeling",
    "relbench.modeling.graph",
    "h5py",
]

for mod in _HEAVY_MODULES:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()
