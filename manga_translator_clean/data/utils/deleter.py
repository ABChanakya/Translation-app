import torch, gc
for obj in ['model', 'optimizer', 'data_loader']:
	if obj in locals():
		del locals()[obj]              # drop big objects you no longer need
gc.collect()                           # run Python garbage collector
torch.cuda.empty_cache()               # tell the allocator to release the cache
