import cv2, queue, threading


class VideoCapture:
    def __init__(self, stream, apiPreference):
        self.cap = cv2.VideoCapture(stream, apiPreference)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                if not self.q.empty():
                    try:
                        self.q.get_nowait()  # discard previous (unprocessed) frame
                    except queue.Empty:
                        pass
                self.q.put((ret, frame))
        except:
            self.cap.release()
            print("Video processing stopped")

    def read(self):
        return self.q.get()

    def release(self):
        return self.cap.release()
