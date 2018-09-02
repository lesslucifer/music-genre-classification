import time


class Timer:
    timers = {}

    @staticmethod
    def start(label, desc = None):
        if (label not in Timer.timers):
            start = time.time()
            Timer.timers[label] = start
            if desc is not None: print(desc)
            # print('%s: %d' % (label, start))

    @staticmethod
    def end(label, desc = None):
        if (label in Timer.timers):
            start = Timer.timers[label]
            end = time.time()
            del Timer.timers[label]
            if desc is not None: print(desc)
            print('%s eslaped %.2f' % (label, end - start))

