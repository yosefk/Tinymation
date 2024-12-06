import viztracer
import shutil
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
        self.nesting_level = 0
        self.tracer = None
        self._event = None
        self.start_time = 0
        self.tracer_stack = []

    def start(self, event):
        class TraceStopper:
            def __enter__(s): pass
            def __exit__(s, *args): self.stop()
        if self.tracer is not None:
            self.tracer_stack.append(self._suspend())
        self.start_time = time.time()
        self._event = event
        if not self.tracer_pool:
            self.tracer = viztracer.VizTracer(ignore_frozen=True)
        else:
            self.tracer = self.tracer_pool.pop()
        self.tracer.start()
        return TraceStopper()

    def stop(self):
        self.tracer.stop()
        total = time.time() - self.start_time
        if total > self.event2slowest.get(self._event, 0):
            self.event2slowest[self._event] = total
            if self._event in self.event2tracer:
                old_tracer = self.event2tracer[self._event]
                old_tracer.clear()
                self.tracer_pool.append(old_tracer)
            self.event2tracer[self._event] = self.tracer
        else:
            self.tracer.clear()
            self.tracer_pool.append(self.tracer)
        self.tracer = None
        if self.tracer_stack:
            self._resume(self.tracer_stack.pop())

    def context(self, context):
        '''we want to aggregate pen and paint bucket events separately, as an example;
        so we call trace.start('mouse-down') and then trace.context('pen') or 'bucket'
        to add to the event name down the road, when the respective code is reached'''
        self._event = context + '.' + self._event

    def event(self, event):
        '''sometimes instead of adding context to an event ("this is a mouse-down event
        that hit the timeline area"), we want to just set it ("this is an undo event,
        we don't care if it came from a keyboard or a mouse event")'''
        self._event = event

    def class_context(self, obj): self.context(obj.__class__.__name__)

    def _suspend(self):
        '''when we have nested events - eg we are loading a clip and periodically running the event
        loop to redraw / show progress - we suspend the current tracer and restart it once we're back to work
        (if we don't do this, nested events are ignored and we get warning messages at stdout)'''
        tracer = self.tracer
        if tracer is not None:
            tracer.stop()
        self.tracer = None
        return (tracer, self._event, self.start_time)

    def _resume(self, context):
        self.tracer, self._event, self.start_time = context
        if self.tracer is not None:
            self.tracer.start()

    def save(self, dir):
        try:
            shutil.rmtree(dir)
        except:
            pass
        try:
            os.makedirs(dir)
        except:
            pass
        with open(os.path.join(dir, 'slowest.txt'), 'w') as f:
            for event, slowest in reversed(sorted(self.event2slowest.items(), key=lambda t: t[1])):
                f.write(f'{slowest*1000:.2f} {event}\n')
        for event, tracer in self.event2tracer.items():
            # prints "Loading finish", apparently from C code, despite verbose=0
            tracer.save(os.path.join(dir, event+'.json'), verbose=0)

trace = Trace()
