#!/usr/bin/python3
'''
we use a pipe to talk to the program under test, rather than sending OS events, for 2 reasons:

* we need a way to ask the program for things, such as the content of the screen or its internal state,
  or where on the screen the elements are that we want to access

* cross-platform frameworks that send OS events directly don't support tablet events, only mouse events
'''

from multiprocessing import Process, Pipe

def talk(conn, command):
    conn.send(command)
    return conn.recv()

def tester_process(conn):
    n = 10
    t = 0
    print(talk(conn,('tablet-press', n*10, 2*n*10, n/10, t)))
    while n:
        print(talk(conn,('tablet-move', n*10, 2*n*10, n/10, t)))
        n -= 1
        t += 7
    print(talk(conn,('tablet-release', n*10, 2*n*10, n/10, t)))

    conn.send('shutdown')

def start_testing():
    parent_conn, child_conn = Pipe()

    tester = Process(target=tester_process, args=(child_conn,))
    tester.start()

    return parent_conn


