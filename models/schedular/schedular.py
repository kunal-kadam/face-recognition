import threading
import time
from schedule import run_pending

def run_continuously(frame_interval=5):
    """Continuously run, while executing pending jobs at each
    elapsed time interval.
    @return cease_continuous_run: threading. Event which can
    be set to cease continuous run. Please note that it is
    *intended behavior that run_continuously() does not run
    missed jobs*. For example, if you've registered a job that
    should run every minute and you set a continuous run
    interval of one hour then your job won't be run 60 times
    at each interval but only once.
    """
    cease_continuous_run_frames = threading.Event()

    class ScheduleThread(threading.Thread):
        @classmethod
        def run(cls):
            while not cease_continuous_run_frames.is_set():
                run_pending()
                time.sleep(frame_interval)

    continuous_thread = ScheduleThread()
    continuous_thread.start()
    return cease_continuous_run_frames
