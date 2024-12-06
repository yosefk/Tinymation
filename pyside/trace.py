import viztracer
import time
import os

class Trace:
    '''collect trace data per event type in start()/stop(); save the
    trace of the slowest event per type in save()'''
    def __init__(self):
        self.event2tracer = {}
        self.event2slowest = {}
        # we reuse tracer objects since apparently not doing so leaks some system resource,
        # as evidenced by the following error message upon the process abruptly exiting:
        #   Failed to create Tss_Key: Resource temporarily unavailable
        self.tracer_pool = []

    def start(self, event):
        self.start_time = time.time()
        self.event = event
        if not self.tracer_pool:
            self.tracer = viztracer.VizTracer(ignore_frozen=True)
        else:
            self.tracer = self.tracer_pool.pop()
        self.tracer.start()

    def stop(self):
        self.tracer.stop()
        total = time.time() - self.start_time
        if total > self.event2slowest.get(self.event, 0):
            self.event2slowest[self.event] = total
            if self.event in self.event2tracer:
                old_tracer = self.event2tracer[self.event]
                old_tracer.clear()
                self.tracer_pool.append(old_tracer)
            self.event2tracer[self.event] = self.tracer
        else:
            self.tracer.clear()
            self.tracer_pool.append(self.tracer)
        self.tracer = None

    def context(self, context):
        '''we want to aggregate pen and paint bucket events separately, as an example;
        so we call trace.start('mouse-down') and then trace.context('pen') or 'bucket'
        to add to the event name down the road, when the respective code is reached'''
        self.event = context + '.' + self.event

    def class_context(self, obj): self.context(obj.__class__.__name__)

    def save(self, dir):
        with open(os.path.join(dir, 'slowest.txt'), 'w') as f:
            for event, slowest in reversed(sorted(self.event2slowest.items(), key=lambda t: t[1])):
                f.write(f'{slowest*1000:.2f} {event}\n')
        for event, tracer in self.event2tracer.items():
            tracer.save(os.path.join(dir, event+'.json'))
        print('execution traces saved to',dir)

trace = Trace()
