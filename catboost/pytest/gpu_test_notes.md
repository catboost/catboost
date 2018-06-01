Test were canonized on TeslaK40
Result could be different on other GPUs
Test should be run in single thread, otherwise could be OOM exception (CatBoost allocates all GPU ram on the start)
