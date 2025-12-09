import queue
import threading
from concurrent.futures import ThreadPoolExecutor


class TaskQueue:
    def __init__(self, max_workers=10):
        """
        Initialize a producer-consumer queue with a thread pool.
        初始化一个生产者-消费者队列，并使用线程池处理任务。

        Args:
            max_workers (int): Maximum number of worker threads.
                               最大线程数。
        """
        self.task_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures = []
        self._futures_lock = threading.Lock()

        # Start background consumer thread
        # 启动后台消费者线程
        self._consumer_thread = threading.Thread(
            target=self._consumer, daemon=True
        )
        self._consumer_thread.start()

    def _consumer(self):
        """
        Background consumer thread:
        Continuously fetches tasks from the queue and submits them to the thread pool.

        后台消费者线程：
        不断从队列中取出任务并提交给线程池执行。

        Each task is in the form: (fn, args, kwargs)
        每个任务格式为： (函数, 参数列表, 关键字参数)
        """
        while True:
            fn, args, kwargs = self.task_queue.get()
            future = self.executor.submit(fn, *args, **kwargs)

            # Record the future for synchronization in wait_all()
            # 记录 future，用于在 wait_all() 中等待所有任务真正执行完
            with self._futures_lock:
                self._futures.append(future)

            # Mark one queue item as processed
            # 标记队列中的一个任务已完成（队列角度）
            self.task_queue.task_done()

    def submit(self, fn, *args, **kwargs):
        """
        Submit a task to the queue.
        向任务队列提交一个任务。

        Args:
            fn (callable): Task function.
                           任务函数。
            *args: Positional arguments for the function.
                   函数的位置参数。
            **kwargs: Keyword arguments for the function.
                      函数的关键字参数。
        """
        self.task_queue.put((fn, args, kwargs))

    def wait_all(self, callback=None, *cb_args, **cb_kwargs):
        """
        Wait until all tasks have finished execution,
        then optionally run a callback function.

        等待所有任务完全执行完成，
        然后（如果提供）执行一个回调函数。

        Args:
            callback (callable): Function to run after completion.
                                 全部完成后的回调函数。
            *cb_args: Arguments passed to the callback.
                      传给回调函数的位置参数。
            **cb_kwargs: Keyword arguments for the callback.
                         传给回调函数的关键字参数。
        """
        # 1. Wait until all tasks have been taken from the queue
        # 1. 等待队列中的所有任务都被消费者线程取走
        self.task_queue.join()

        # 2. Wait until all submitted futures have completed
        # 2. 等待线程池中所有已提交的任务执行完成
        with self._futures_lock:
            futures_copy = list(self._futures)
            self._futures.clear()

        for f in futures_copy:
            # Block until task finishes; exceptions are re-raised here
            # 阻塞直到任务执行完；任务中出现异常会在这里重新抛出
            f.result()

        # 3. Execute callback if provided
        # 3. 如果有回调函数，则执行
        if callback is not None:
            callback(*cb_args, **cb_kwargs)

# use case
import time

def my_task(x):
    time.sleep(1)
    print(f"Task {x} done")

def all_done_callback(msg):
    print("Callback:", msg)

if __name__ == "__main__":
    q = TaskQueue(max_workers=4)

    # 提交 10 个任务
    for i in range(10):
        q.submit(my_task, i)

    # 等待所有任务完成，然后执行回调函数
    q.wait_all(all_done_callback, "所有任务都处理完了！")
