"""Use the reviewed auditor unchanged, adding the full FSDP replay alias."""
import importlib.util
from pathlib import Path
p=Path(__file__).parent/"root_verify_four_full.py"
s=importlib.util.spec_from_file_location("four_audit",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.LAYOUTS["fsdp_full"]=(4,1,1,1,1)
m.main()
