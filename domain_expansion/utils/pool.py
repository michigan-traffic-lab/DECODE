from time import sleep
from tqdm import trange, tqdm
from random import random
from multiprocessing import Pool, Manager
from itertools import count

def _wrap_func(func, args, pool, nlock):
    n = -1
    with nlock:
        n = next(i for i,v in enumerate(pool) if v == 0)
        pool[n] = 1
    ret = func(n, *args)
    return (n, ret)

class NumberPool:
    def __init__(self, nworkers, *args, **kargs):
        self._ppool = Pool(nworkers, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),), *args, **kargs)
        self._npool = Manager().Array('B', [0] * nworkers)
        self._nlock = Manager().Lock()

    def apply_async(self, func, args=(), callback=None):
        def _wrap_cb(ret):
            n, oret = ret
            with self._nlock:
                self._npool[n] = 0
            if callback:
                callback(oret)
        return self._ppool.apply_async(_wrap_func,
            (func, args, self._npool, self._nlock),
            callback=_wrap_cb,
            error_callback=lambda x: print(x))
    
    def close(self):
        self._ppool.close()

    def join(self):
        self._ppool.join()

def progresser(pos, n):
    interval = random() * 0.001
    total = 5000
    text = "#{}, est. {:<04.2}s".format(n, interval * total)
    for _ in trange(total, desc=text, position=(pos+1), leave=False):
        sleep(interval)
    return n

def callback(n):
    tqdm.write(f"\x1b[2K\r#{n} finished")

if __name__ == '__main__':
    p = NumberPool(8)
    for i in tqdm(range(30), desc="mainloader", position=0, leave=False):
        p.apply_async(progresser, (i,), callback=callback)
        sleep(random()/3)
    p.close()
    p.join()

    print("\x1b[J")
