import torch
import torch_xla
import torch_xla.debug.profiler as xp
import torch_xla.utils.keyd_queue as kq
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm
from torch_xla.distributed.parallel_loader import PerDeviceLoader, PerDeviceQueue
from multiprocessing import Process, Queue, Pool

class ParallelLoader(object):
  def __init__(self,
               loader,
               devices,
               batchdim=0,
               batches_per_execution=1,
               loader_prefetch_size=8,
               device_prefetch_size=4,
               host_to_device_transfer_threads=1,
               num_workers=4):
    self._loader = loader
    self._devices = [torch.device(x) for x in devices]
    self._batchdim = batchdim
    self._batches_per_execution = batches_per_execution
    self._done = False
    self._queues = dict()
    self._num_workers = num_workers

    for device in self._devices:
      self._queues[device] = PerDeviceQueue(device, loader_prefetch_size,
                                            device_prefetch_size)
    self._pool = Pool(num_workers)
    self._loader_process = Process(target=self._loader_worker)
    self._loader_process.daemon = True
    self._loader_process.start()
    for dqueue in self._queues.values():
      for i in range(host_to_device_transfer_threads):
        p = Process(target=self._worker, args=(dqueue,))
        p.daemon = True
        p.start()

  def per_device_loader(self, device):
    return PerDeviceLoader(self, torch.device(device))

  def per_device_samples(self):
    return len(self._loader) // len(self._devices)

  def next_item(self, device):
    dqueue = self._queues[device]
    return dqueue.queue.get()

  def close(self):
    self._done = True
    for dqueue in self._queues.values():
      dqueue.queue.close()
      dqueue.loader_queue.close()
    self._pool.close()

  @property
  def batches_per_execution(self):
    return self._batches_per_execution

  def _loader_worker(self):
    queues = list(self._queues.values())
    data_iter = self._loader
    for batch in self._pool.imap(self._enumerate_and_send, data_iter):
      for queue_no, device_batch in enumerate(batch):
        queues[queue_no].loader_queue.put(device_batch)
    for dqueue in queues:
      dqueue.loader_queue.close_write()

  def _enumerate_and_send(self, item):
    return [(i, data) for i, data in enumerate(item)]

  def _get_batch(self, dqueue):
    batch = []
    while dqueue.queue.max_size() > len(batch):
      item = dqueue.loader_queue.get()
      if item is None:
        break
      batch.append(item)
    return batch

  def _worker(self, dqueue):
    device = torch.device(dqueue.device)
    while True:
      batch = self._get_batch(dqueue)
      if not batch:
        break
      batch = xm.send_cpu_data_to_device(batch, device)
      for data in batch:
        dqueue.queue.put(data)