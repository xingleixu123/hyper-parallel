"""Run the production precision configuration with NS shape-only provenance."""

import capture_training
from ns_batch_trace import install_ns_batch_trace


if __name__ == '__main__':
    install_ns_batch_trace(capture_training.OUTPUT, capture_training.RANK)
    capture_training.main()
