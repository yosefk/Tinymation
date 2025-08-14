try:
    import viztracer
    has_trace = True
except:
    has_trace = False
import collections
import shutil
import time
import os

MAX_TRACER_OBJECTS = 500 # we see "Failed to create Tss_Key" after ~1000 objects,
# but better to be on the safe side since not sure what else creates Tss_Keys...

class EventData:
    def __init__(self):
        self.tracer = None
        self.start = 0
        self.total = 0
        self.sum_total = 0
        self.sum_calls = 0
    def average(self):
        return self.sum_total / self.sum_calls

if has_trace:
  class Trace:
    '''collect trace data per event type in start()/stop(); save the
    trace of the slowest event per type in save()'''
    def __init__(self):
        self.event2data = collections.OrderedDict()
        # we reuse tracer objects since apparently not doing so leaks some system resource,
        # as evidenced by the following error message upon the process abruptly exiting:
        #   Failed to create Tss_Key: Resource temporarily unavailable
        self.tracer_pool = []
        self.nesting_level = 0
        self.tracer = None
        self._event = None
        self.start_time = 0
        self.tracer_stack = []

    def clear(self):
        if self.tracer is not None:
            self.tracer.stop()
        self.nesting_level = 0
        self.tracer = None
        self._event = None
        self.start_time = 0
        self.tracer_stack = []
        for data in self.event2data.values():
            if data.tracer is not None:
                data.tracer.clear()
                self.tracer_pool.append(data.tracer)
        self.event2data = collections.OrderedDict()

    def start(self, event):
        '''with trace.start('event-name'):
            code()
        ...will trace the code in the with  block
        '''
        class TraceStopper:
            def __enter__(s): pass
            def __exit__(s, *args): self.stop()
        if self.tracer is not None:
            self.tracer_stack.append(self._suspend())
        self._event = event
        if not self.tracer_pool:
            if len(self.event2data) >= MAX_TRACER_OBJECTS:
                self.event2data.popitem(last=False) # LRU eviction...
                # better than process termination upon failing to create Tss_Key
            self.tracer = viztracer.VizTracer(ignore_frozen=True)
        else:
            self.tracer = self.tracer_pool.pop()
        self.start_time = time.time()
        self.tracer.start()
        return TraceStopper()

    def stop(self):
        '''better use with trace.start(event): rather than call stop directly'''
        if self.tracer is None:
            return
        self.tracer.stop()
        total = time.time() - self.start_time
        data = self.event2data.setdefault(self._event, EventData())
        data.sum_total += total
        data.sum_calls += 1
        if total > data.total:
            data.start = self.start_time
            data.total = total
            if data.tracer is not None:
                data.tracer.clear()
                self.tracer_pool.append(data.tracer)
            data.tracer = self.tracer
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
        '''saves a trace json file per event (the slowest trace for that event),
        and a slowest.txt report showing the worst case latency of each event and when it was observed'''
        try:
            shutil.rmtree(dir)
        except:
            pass
        try:
            os.makedirs(dir)
        except:
            pass
        if not os.path.isdir(dir):
            print('warning: could not create trace directory',dir)
            return
        with open(os.path.join(dir, 'slowest.txt'), 'w') as f:
            for event, data in reversed(sorted(self.event2data.items(), key=lambda t: t[1].total)):
                start = time.strftime('%H:%M:%S', time.localtime(data.start))
                f.write(f'{data.total*1000:.2f} {event} {start} {data.average()*1000:.2f}\n')
        for event, data in self.event2data.items():
            # prints "Loading finish", apparently from C code, despite verbose=0
            if data.tracer is not None:
                data.tracer.save(os.path.join(dir, event+'.json'), verbose=0)

else: # no viztracer
  class Trace:
    def __init__(self): pass
    def clear(self): pass
    def start(self, event):
        class TraceStopper:
            def __enter__(s): pass
            def __exit__(s, *args): pass
        return TraceStopper()
    def stop(self): pass
    def context(self, context): pass
    def event(self, event): pass
    def class_context(self, obj): pass 
    def save(self, dir): pass

trace = Trace()

# tests
#######

# we test that we don't create more than MAX_TRACER_OBJECTS events, and that nesting trace.start() calls works

def test_nested_tracing(tmp_path):
    global MAX_TRACER_OBJECTS
    MAX_TRACER_OBJECTS = 20

    for i in range(10):
        with trace.start('outer%d'%i):
            for j in range(10):
                with trace.start('inner%d'%j):
                    pass

    trace.save(str(tmp_path))

def test_invalid_dir():
    test_nested_tracing('/not/found')
