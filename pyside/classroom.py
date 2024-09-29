
# Student server & teacher client: turn screen on/off, save/restore backups

#import threading
#import socket
#from zeroconf import ServiceInfo, Zeroconf
#from http.server import BaseHTTPRequestHandler, HTTPServer
#import getpass, getmac
#import base64

class BaseHTTPRequestHandler: pass
class StudentRequestHandler(BaseHTTPRequestHandler):
    def do_PUT(self):
        try:
            if not self.path.startswith('/put/'):
                raise Exception("bad PUT path")
            fname = os.path.join(WD, self.path[len('/put/'):])
            if not os.path.exists(fname):
                size = int(self.headers['Content-Length'])
                data = self.rfile.read(size)
                with open(fname, 'wb') as f:
                    f.write(data)

                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(b'PUT request processed successfully')
                return
        except:
            import traceback
            traceback.print_exc()

        self.send_response(500)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(bytes(f'failed to handle PUT path: {self.path}', 'utf-8'))

    def do_GET(self):
        response = 404
        message = f'Unknown path: {self.path}'

        try:
            if self.path == '/lock':
                student_server.lock_screen = True
                pg.event.post(pg.Event(REDRAW_LAYOUT_EVENT))
                message = 'Screen locked'
                response = 200
            elif self.path == '/unlock':
                student_server.lock_screen = False
                pg.event.post(pg.Event(REDRAW_LAYOUT_EVENT))
                message = 'Screen unlocked'
                response = 200
            elif self.path == '/drawing_layout':
                layout.mode = DRAWING_LAYOUT
                pg.event.post(pg.Event(REDRAW_LAYOUT_EVENT))
                message = 'Layout set to drawing'
                response = 200
            elif self.path == '/animation_layout':
                layout.mode = ANIMATION_LAYOUT
                pg.event.post(pg.Event(REDRAW_LAYOUT_EVENT))
                message = 'Layout set to animation'
                response = 200
            elif self.path == '/mac':
                message = str(getmac.get_mac_address())
                response = 200
            elif self.path == '/backup':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()

                def on_progress(compressed, total):
                    self.wfile.write(bytes(f'{compressed} {total} <br>\n', "utf8"))
                abspath = create_backup(on_progress)

                backup_props = {}
                backup_props['size'] = os.path.getsize(abspath)
                
                user = getpass.getuser()
                host = student_server.host
                mac = getmac.get_mac_address()

                # we deliberately rename here and not on the client since if a computer keeps the files
                # across sessions, it will save us a transfer when restoring the backup to have the file
                # already stored with the name the server will use to restore it
                file = os.path.basename(abspath)
                fname = f'student-{user}@{host}-{mac}-{file}'.replace(':','_')
                shutil.move(abspath, os.path.join(os.path.dirname(abspath), fname))

                backup_props['file'] = fname

                message = json.dumps(backup_props)
                self.wfile.write(bytes(message, "utf8"))
                return
            elif self.path.startswith('/file/'):
                fpath = self.path[len('/file/'):]
                fpath = os.path.join(WD, fpath)
                if os.path.exists(fpath):
                    response = 200
                    with open(fpath, 'rb') as f:
                        data = f.read()
                    data64 = base64.b64encode(data)
                    self.send_response(response)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()
                    chunk = 64*1024
                    for i in range(0, len(data64), chunk):
                        self.wfile.write(data64[i:i+chunk]+b'\n')
                    return
            elif self.path.startswith('/unzip/'):
                fpath = self.path[len('/unzip/'):]
                fpath = os.path.join(WD, fpath)
                if os.path.exists(fpath):
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html')
                    self.end_headers()

                    def on_progress(uncompressed, total):
                        self.wfile.write(bytes(f'{uncompressed} {total} <br>\n', "utf8"))
                    unzip_files(fpath, on_progress)
                    pg.event.post(pg.Event(RELOAD_MOVIE_LIST_EVENT))
                    return

        except Exception:
            import traceback
            traceback.print_exc()
            message = f'internal error handling {self.path}'
            response = 500

        self.send_response(response)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(bytes(message, "utf8"))

class StudentServer:
    def __init__(self):
        self.lock_screen = False

        self.host = socket.gethostname()
        self.host_addr = socket.gethostbyname(self.host)
        self.port = 8080
        self.zeroconf = Zeroconf()
        self.service_info = ServiceInfo(
            "_http._tcp.local.",
            f"Tinymation.{self.host}.{self.host_addr}._http._tcp.local.",
            addresses=[socket.inet_aton(self.host_addr)],
            port=self.port)
        self.zeroconf.register_service(self.service_info)
        print(f"Student server running on {self.host}[{self.host_addr}]:{self.port}")

        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def _run(self):
        server_address = ('', self.port)
        self.httpd = HTTPServer(server_address, StudentRequestHandler)
        self.httpd.serve_forever()

    def stop(self):
        self.httpd.shutdown()
        self.thread.join()
        self.zeroconf.unregister_service(self.service_info)

#student_server = StudentServer()

#from zeroconf import ServiceBrowser
#import http.client

class StudentThreads:
    def __init__(self, students, title):
        self.threads = []
        self.done = []
        self.students = students
        self.student2progress = {}
        self.progress_bar = ProgressBar(title)

    def run_thread_func(self, student, conn):
        try:
            self.student_thread(student, conn)
        except:
            import traceback
            traceback.print_exc()
        finally:
            self.done.append(student)
            conn.close()

    def thread_func(self, student, conn):
        def thread():
            return self.run_thread_func(student, conn)
        return thread

    def student_thread(self, student, conn): pass 

    def start_thread_per_student(self):
        for student in self.students:
            host, port = self.students[student]
            conn = http.client.HTTPConnection(host, port)

            thread = threading.Thread(target=self.thread_func(student, conn))
            thread.start()
            self.threads.append(thread)

    def wait_for_all_threads(self):
        while len(self.done) < len(self.students):
            progress = self.student2progress.copy().values()
            done = sum([p[0] for p in progress])
            total = max(1, sum([p[1] for p in progress]))
            self.progress_bar.on_progress(done, total)
            time.sleep(0.3)

        for thread in self.threads:
            thread.join()

class TeacherClient:
    def __init__(self):
        self.students = {}
        self.screens_locked = False
    
        self.zeroconf = Zeroconf()
        self.browser = ServiceBrowser(self.zeroconf, "_http._tcp.local.", self)

    def remove_service(self, zeroconf, type, name):
        if name in self.students:
            del self.students[name]
            pg.event.post(pg.Event(REDRAW_LAYOUT_EVENT))

    def add_service(self, zeroconf, type, name):
        if name.startswith('Tinymation'):
            info = zeroconf.get_service_info(type, name)
            if info:
                host, port = socket.inet_ntoa(info.addresses[0]), info.port
                self.students[name] = (host, port)
                pg.event.post(pg.Event(REDRAW_LAYOUT_EVENT))

                # if the screens are supposed to be locked and a student restarted the program or just came
                # and started it, we want the screen locked, even at the cost of interrupting whatever
                # the teacher is doing. if on the other hand the student screens are locked and the teacher
                # restarted the program (a rare event), the teacher can unlock explicitly and will most
                # certainly do so, so we don't want to have a bunch of unlocking happen automatically
                # as the teacher's program discovers the live student programs.
                if self.screens_locked:
                    self.broadcast_request('/lock', 'Locking 1...', [name])
                    pg.event.post(pg.Event(REDRAW_LAYOUT_EVENT))

                if layout.mode == DRAWING_LAYOUT:
                    self.broadcast_request('/drawing_layout', 'Drawing layout 1...', [name])
                    pg.event.post(pg.Event(REDRAW_LAYOUT_EVENT))

    def update_service(self, zeroconf, type, name):
        num_students = len(self.students)
        if name.startswith('Tinymation'):
            info = zeroconf.get_service_info(type, name)
            if info:
                host, port = socket.inet_ntoa(info.addresses[0]), info.port
                if (host, port) != self.students[name]:
                    self.students[name] = (host, port)
            else:
                del self.students[name]
        if num_students != len(self.students):
            pg.event.post(pg.Event(REDRAW_LAYOUT_EVENT))

    def send_request(self, student, url):
        host, port = self.students[student]
        conn = http.client.HTTPConnection(host, port)
        headers = {'Content-Type': 'text/html'}
        conn.request('GET', url, headers=headers)
        
        response = conn.getresponse()
        status = response.status
        message = response.read().decode()
        conn.close()
        return status, message

    def broadcast_request(self, url, progress_bar_title, students):
        # a big reason for the progress bar is, when a student computer hybernates [for example],
        # remove_service isn't called, and it takes a while to reach a timeout. TODO: see what needs
        # to be done to improve student machine hybernation and waking up from it
        progress_bar = ProgressBar(progress_bar_title)
        responses = {}
        for i, student in enumerate(students):
            responses[student] = self.send_request(student, url)
            progress_bar.on_progress(i+1, len(students))
        return responses

    # locking and unlocking deliberately locks up the teacher's main thread - you want to know the students'
    # screen state, eg you don't want to keep going when some of their screens aren't locked
    def lock_screens(self):
        students = self.students.keys()
        self.broadcast_request('/lock', f'Locking {len(students)}...', students)
        self.screens_locked = True
    def unlock_screens(self):
        students = self.students.keys()
        self.broadcast_request('/unlock', f'Unlocking {len(students)}...', students)
        self.screens_locked = False

    def drawing_layout(self):
        students = self.students.keys()
        self.broadcast_request('/drawing_layout', f'Drawing layout {len(students)}...', students)
    def animation_layout(self):
        students = self.students.keys()
        self.broadcast_request('/animation_layout', f'Drawing layout {len(students)}...', students)

    def get_backup_info(self, students):
        backup_info = {}

        class BackupInfoStudentThreads(StudentThreads):
            def student_thread(self, student, conn):
                headers = {'Content-Type': 'text/html'}
                conn.request('GET', '/backup', headers=headers)
                response = conn.getresponse()
                while True:
                    line = response.fp.readline().decode('utf-8').strip()
                    if not line:
                        break
                    print(student, line)
                    if line.endswith('<br>'):
                        self.student2progress[student] = [int(t) for t in line.split()[:2]]
                    else:
                        backup_info[student] = json.loads(line)
                        break

        student_threads = BackupInfoStudentThreads(students, f'Saving {len(students)}+1...')
        student_threads.start_thread_per_student()

        def my_backup_thread():
            teacher_id = None
            def on_progress(compressed, total):
                student_threads.student2progress[teacher_id] = (compressed, total)
            filename = create_backup(on_progress)
            backup_info[teacher_id] = {'file':filename}

        teacher_thread = threading.Thread(target=my_backup_thread)
        teacher_thread.start()

        student_threads.wait_for_all_threads()
        teacher_thread.join()

        return backup_info

    def get_backups(self, backup_info, students):
        backup_dir = os.path.join(WD, f'Tinymation-class-backup-{format_now()}')
        try:
            os.makedirs(backup_dir)
        except:
            pass

        class BackupInfoStudentThreads(StudentThreads):
            def student_thread(self, student, conn):
                headers = {'Content-Type': 'text/html'}
                conn.request('GET', '/file/'+backup_info[student]['file'], headers=headers)

                response = conn.getresponse()
                backup_base64 = b''
                total = backup_info[student]['size']
                while True:
                    line = response.fp.readline()
                    if not line:
                        break
                    self.student2progress[student] = (len(backup_base64)*5/8, total)
                    backup_base64 += line
                    print(student, 'sent', int(len(backup_base64)*5/8), '/', total)

                data = base64.b64decode(backup_base64)
                info = backup_info[student]
                file = info['file']
                with open(os.path.join(backup_dir, file), 'wb') as f:
                    f.write(data)

                self.student2progress[student] = (total, total)
    
        student_threads = BackupInfoStudentThreads(students, f'Receiving {len(students)}...')
        student_threads.start_thread_per_student()

        teacher_id = None
        teacher_file = backup_info[teacher_id]['file']
        target_teacher_file = os.path.join(backup_dir, 'teacher-'+os.path.basename(teacher_file))
        shutil.move(teacher_file, target_teacher_file)

        student_threads.wait_for_all_threads()
        
        open_explorer(backup_dir)

    def save_class_backup(self):
        students = self.students.copy() # if someone connects past this point, we don't have their backup
        student_backups = self.get_backup_info(students)
        self.get_backups(student_backups, students)

    def restore_class_backup(self, class_backup_dir):
        if not class_backup_dir:
            return
        # ATM restores to machines based on their MAC addresses. we could also have a mode where we
        # just restore to arbitrary machines (5 backups, 6 machines - pick 5 random ones.) this could
        # be the right thing if the machines in the class are a different set every time and they
        # all erase their files. this would be more trouble if at least some machines kept the files
        # and some didn't, since given our reluctance to remove or rename existing clips, we could
        # create directories with a mix of clips from different students. if a subset of machines keep
        # their files and are the same as when the backup was made, our system of assigning by MAC
        # works well since the ones deleting the files will get them back and the rest will be unaffected
        #
        # note that we don't "restore" as in "go back in time" - if some of the clips were edited
        # after the backup was made, we don't undo these changes, since we never overwrite existing
        # files. [we can create "orphan" files this way if a frame was deleted... we would "restore" its
        # images but would not touch the movie metadata. this seems harmless enough]
        students = self.students.copy()
        responses = self.broadcast_request('/mac', 'Getting IDs...', students)
        student_macs = dict([(student, r.replace(':','_')) for (student, (s,r)) in responses.items()])
        backup_files = [os.path.join(class_backup_dir, f) for f in os.listdir(class_backup_dir)]

        student2backup = {}
        teacher_backup = None
        for student,mac in student_macs.items():
            for f in backup_files:
                if mac in f:
                    student2backup[student] = f
                    break
        for f in backup_files:
            if 'teacher-' in f:
                teacher_backup = f

        def my_backup_thread():
            if not teacher_backup:
                return
            def on_progress(uncompressed, total): pass
            unzip_files(teacher_backup, on_progress)
            pg.event.post(pg.Event(RELOAD_MOVIE_LIST_EVENT))

        teacher_thread = threading.Thread(target=my_backup_thread)
        teacher_thread.start()

        self.put(student2backup, students)
        self.unzip(student2backup, students)

        teacher_thread.join()

    def put(self, student2file, students):
        class PutThreads(StudentThreads):
            def student_thread(self, student, conn):
                if student not in student2file:
                    return
                file = student2file[student]
                with open(file, 'rb') as f:
                    data = f.read()

                conn.putrequest('PUT', '/put/'+os.path.basename(file))
                conn.putheader('Content-Length', str(len(data)))
                conn.putheader('Content-Type', 'application/octet-stream')
                conn.endheaders()

                chunk_size = 64*1024
                for i in range(0, len(data), chunk_size):
                    conn.send(data[i:i+chunk_size])
                    self.student2progress[student] = i, len(data)
                self.student2progress[student] = len(data), len(data)

                response = conn.getresponse()
                response.read()

        student_threads = PutThreads(students, f'Sending {len(students)}...')
        student_threads.start_thread_per_student()
        student_threads.wait_for_all_threads()

    def unzip(self, student2file, students):
        class UnzipThreads(StudentThreads):
            def student_thread(self, student, conn):
                if student not in student2file:
                    return
                zip_file = student2file[student]
                headers = {'Content-Type': 'text/html'}
                conn.request('GET', '/unzip/'+os.path.basename(zip_file), headers=headers)
                response = conn.getresponse()
                while True:
                    line = response.fp.readline().decode('utf-8').strip()
                    if not line:
                        break
                    if line.endswith('<br>'):
                        self.student2progress[student] = [int(t) for t in line.split()[:2]]

        student_threads = UnzipThreads(students, f'Unzipping {len(students)}...')
        student_threads.start_thread_per_student()
        student_threads.wait_for_all_threads()

    def put_dir(self, dir):
        if not dir:
            return
        dir = os.path.realpath(dir)
        progress_bar = ProgressBar('Zipping...')
        zip_file = os.path.join(WD, dir + '.zip')
        zip_dir(zip_file, dir, progress_bar.on_progress, os.path.dirname(dir))
        students = self.students.copy()
        student2file = dict([(student, zip_file) for student in students])
        self.put(student2file, students)
        self.unzip(student2file, students)
        os.unlink(zip_file)

teacher_client = None

class Layout:
    def draw_students(self):
        if teacher_client:
            text_surface = font.render(f"{len(teacher_client.students)} students", True, (255, 0, 0), (255, 255, 255))
            screen.blit(text_surface, ((screen.get_width()-text_surface.get_width()), (screen.get_height()-text_surface.get_height())))


def start_teacher_client():
    global student_server
    global teacher_client

    student_server.stop()
    student_server = None
    teacher_client = TeacherClient()


def process_keydown_event():

    # teacher/student - TODO: better UI
    # Ctrl-T: teacher client
    if ctrl and event.key() == Qt.Key_T:
        print('shutting down the student server and starting the teacher client')
        start_teacher_client()
        return
    if ctrl and event.key() == Qt.Key_L and teacher_client:
        print('locking student screens')
        teacher_client.lock_screens()
        return
    if ctrl and event.key() == Qt.Key_U and teacher_client:
        print('unlocking student screens')
        teacher_client.unlock_screens()
        return
    if ctrl and event.key() == Qt.Key_B and teacher_client:
        print('saving class backup')
        teacher_client.save_class_backup()
        return
    if ctrl and event.key() == Qt.Key_D and teacher_client:
        print('restoring class backup')
        teacher_client.restore_class_backup(open_dir_path_dialog())
        return
    if ctrl and event.key() == Qt.Key_P and teacher_client:
        print("putting a directory in all students' Tinymation directories")
        teacher_client.put_dir(open_dir_path_dialog())
        return

    # Ctrl-1/2: set layout to drawing/animation
    if ctrl and event.key() == Qt.Key_1:
        layout.mode = DRAWING_LAYOUT
        if teacher_client:
            teacher_client.drawing_layout()
        return
    if ctrl and event.key() == Qt.Key_2:
        layout.mode = ANIMATION_LAYOUT
        if teacher_client:
            teacher_client.animation_layout()
        return

class ScreenLock:
    def __init__(self):
        self.locked = False

    def is_locked(self):
        if student_server is not None and student_server.lock_screen:
            layout.draw_locked()
            pygame.display.flip()
            try_set_cursor(empty_cursor)
            self.locked = True
            return True
        elif self.locked:
            layout.draw()
            pygame.display.flip()
            try_set_cursor(layout.full_tool.cursor[0])
            self.locked = False
        return False

class ScreenLock:
    def __init__(self):
        self.locked = False

    def is_locked(self):
        if student_server is not None and student_server.lock_screen:
            layout.draw_locked()
            pygame.display.flip()
            try_set_cursor(empty_cursor)
            self.locked = True
            return True
        elif self.locked:
            layout.draw()
            pygame.display.flip()
            try_set_cursor(layout.full_tool.cursor[0])
            self.locked = False
        return False

